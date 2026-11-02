"""Streamlit interaction tests, with controlled backend results and no API calls."""

from __future__ import annotations

import importlib.util
import queue
import sys
import threading
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

from downstream import DownstreamError, DownstreamResult, EvidenceSource
from orchestration.types import FinalResult, QueryPlan, ReviewRecord
from tests.test_downstream import passage, search_result

APP = str(Path(__file__).resolve().parents[1] / "ui" / "app.py")


def lookup_store(passages):
    """Patch for retrieval.lookup.get_lookup_store serving the given passage records."""
    store = MagicMock()
    store.get_passages.return_value = passages
    return patch("retrieval.lookup.get_lookup_store", return_value=store)


class UITests(unittest.TestCase):
    def setUp(self):
        # The page warms up retrieval in the background; never load real models or indexes here.
        warm_up = patch("retrieval.pipeline.retrieve")
        warm_up.start()
        self.addCleanup(warm_up.stop)
        self.search = search_result()
        self.source = EvidenceSource(1, "2401.00001", "P2401.00001_0000", "Paper 2401.00001", 2024,
                                     "Method", "https://arxiv.org/abs/2401.00001v2", passage()["text"])
        self.answer = DownstreamResult("q-test", "How does it work?", "Low-rank updates W[0] [1].", "answered",
                                       [self.source], [1], self.search,
                                       {"total_ms": 100, "generation_ms": 20})

    def test_submit_reuses_search_and_shows_citation_and_original_text(self):
        with patch("downstream.run", return_value=self.answer) as answer, \
                patch("orchestration.run_query") as search, \
                lookup_store({self.source.passage_id: passage()}):
            app = AppTest.from_file(APP).run()
            app.text_area[0].input("How does it work?")
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            answer.assert_called_once()
            self.assertEqual(answer.call_args.args, ("How does it work?",))
            self.assertTrue(callable(answer.call_args.kwargs["progress"]))
            search.assert_not_called()
            self.assertTrue(app.status[0].label.startswith("Answer ready in "))
            self.assertEqual([tab.label for tab in app.tabs], ["Paper QA", "Paper Search"])
            # Notation stays unlinked; citations keep their brackets, so [1][2] never reads as 12.
            self.assertTrue(any("W[0] [\\[1\\]](#source-1)" in item.value for item in app.markdown))
            self.assertTrue(any(item.value == self.source.text for item in app.text))
            self.assertTrue(any("[1] Paper" in item.label for item in app.expander))
            self.assertEqual([button.proto.label for button in app.get("download_button")],
                             ["Download answer and sources (JSON)", "Download search record (JSON)"])
            app.run()
            answer.assert_called_once()

    def test_failed_followup_clears_stale_answer(self):
        with patch("downstream.run", side_effect=[self.answer, DownstreamError("Paper search failed.")]), \
                lookup_store({}):
            app = AppTest.from_file(APP).run()
            app.text_area[0].input("First question")
            app.button[0].click().run()
            app.text_area[0].input("Second question")
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.error[0].value, "Paper search failed.")
            self.assertFalse(any("Low-rank updates" in item.value for item in app.markdown))

    def test_empty_input_and_example_selection_do_not_call_backend(self):
        with patch("downstream.run") as answer, patch("orchestration.run_query") as search:
            app = AppTest.from_file(APP).run()
            app.button[0].click().run()
            self.assertEqual(app.warning[0].value, "Enter a question first.")
            example = app.button(key="example_1").label
            app.button(key="example_1").click().run()
            self.assertEqual(app.text_area[0].value, example)
            answer.assert_not_called()
            search.assert_not_called()

    def test_search_only_does_not_generate_answer(self):
        with patch("downstream.run") as answer, patch("orchestration.run_query", return_value=self.search) as search, \
                lookup_store({}):
            app = AppTest.from_file(APP).run()
            app.text_area[0].input("Find papers")
            app.button[1].click().run()
            self.assertEqual(len(app.exception), 0)
            search.assert_called_once()
            self.assertEqual(search.call_args.args, ("Find papers",))
            answer.assert_not_called()
            self.assertTrue(app.status[0].label.endswith(": 1 paper accepted"))
            self.assertEqual(app.session_state["tab"], "Paper Search")  # its tab opens, not Paper QA's

    def test_steps_show_while_running_and_stay_after_a_rerun(self):
        def run_with_steps(question, progress):
            progress("understanding", {})
            progress("planned", {"queries": 3, "aspects": 2})
            progress("round", {"round": 1, "queries": 3})
            progress("round_done", {"round": 1, "reviewed": 12, "passed": 1, "covered": 1, "aspects": 2})
            progress("follow_up", {"missing": ["Training cost."]})
            progress("answering", {"sources": 1})
            return self.answer

        with patch("downstream.run", side_effect=run_with_steps), \
                lookup_store({self.source.passage_id: passage()}):
            app = AppTest.from_file(APP).run()
            app.text_area[0].input("How does it work?")
            app.button[0].click().run()
            app.run()  # a later rerun (opening an excerpt, a download) keeps the finished steps
            self.assertEqual(len(app.exception), 0)
            status = app.status[0]
            self.assertTrue(status.label.startswith("Answer ready in "))
            self.assertEqual(status.markdown[0].value.split("\n\n"), [
                "Reading the question and planning the searches.",
                "Planned 3 searches for 2 aspects of the question.",
                "Round 1: running 3 searches.",
                "Round 1 done: 12 papers reviewed so far, 1 accepted, 1 of 2 aspects covered.",
                "Not covered yet: Training cost. Writing new searches for it.",
                "Writing the answer from 1 excerpt.",
            ])

    def test_no_answer_says_why_and_shows_the_closest_rejected_papers(self):
        search = FinalResult("q-none", "Protein folding?", QueryPlan("Protein folding?"), [],
                             {"rounds": 3, "papers_reviewed": 9, "termination_reason": "max_rounds"},
                             reviews=[ReviewRecord("2401.00002", "reject", 2, reasons="About text, not proteins.",
                                                   evidence={"title": "A text paper"})])
        empty = DownstreamResult("q-none", "Protein folding?", "No sufficient paper evidence was found.",
                                 "insufficient_evidence", [], [], search, {"total_ms": 100})
        with patch("downstream.run", return_value=empty), lookup_store({}):
            app = AppTest.from_file(APP).run()
            app.text_area[0].input("Protein folding?")
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn("no paper covers this question", app.status[0].label)
            self.assertIn("The search ran 3 rounds and the reviewer read 9 papers", app.info[0].value)
            self.assertTrue(any("turned down (1)" in item.label for item in app.expander))
            self.assertEqual(app.expander[0].label, "Examples")  # folded once there is a result
            self.assertTrue(any(item.value == "About text, not proteins." for item in app.caption))


    def running_request(self, future, lines=(), events=(), cancel=None):
        """A request as `_start` leaves it in session state, for a rerun to pick up."""
        waiting = queue.Queue()
        for event in events:
            waiting.put(event)
        return SimpleNamespace(ask=True, future=future, events=waiting, cancel=cancel or threading.Event(),
                               started=time.perf_counter() - 5, lines=list(lines))

    def test_a_rerun_during_a_request_picks_it_up_and_shows_its_result(self):
        # A tab or example click reruns the script while a request runs; it must not drop it.
        done = Future()
        done.set_result(self.answer)
        with patch("downstream.run") as answer, lookup_store({self.source.passage_id: passage()}):
            app = AppTest.from_file(APP).run()
            app.session_state["request"] = self.running_request(
                done, lines=["Reading the question and planning the searches."],
                events=[("answering", {"sources": 1})])
            app.run()  # no submit: the rerun of a tab click
            self.assertEqual(len(app.exception), 0)
            answer.assert_not_called()  # followed, not started again
            self.assertTrue(app.status[0].label.startswith("Answer ready in "))
            self.assertEqual(app.status[0].markdown[0].value.split("\n\n"), [
                "Reading the question and planning the searches.", "Writing the answer from 1 excerpt."])
            self.assertTrue(any("W[0] [\\[1\\]](#source-1)" in item.value for item in app.markdown))
            self.assertNotIn("request", app.session_state)

    def test_a_new_question_cancels_the_running_one(self):
        cancel = threading.Event()
        with patch("downstream.run", return_value=self.answer) as answer, \
                lookup_store({self.source.passage_id: passage()}):
            app = AppTest.from_file(APP).run()
            app.session_state["request"] = self.running_request(Future(), cancel=cancel)  # never finishes
            app.text_area[0].input("How does it work?")
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(cancel.is_set())
            answer.assert_called_once()
            self.assertTrue(app.status[0].label.startswith("Answer ready in "))

    def test_a_cancelled_request_stops_at_its_next_step(self):
        spec = importlib.util.spec_from_file_location("paperodyssey_ui_app", APP)
        ui = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = ui  # dataclasses look the module up while it is being created
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(ui)
        release, after = threading.Event(), []

        def slow_run(question, progress):
            progress("understanding", {})
            release.wait(5)
            progress("planned", {"queries": 1, "aspects": 1})  # raises once cancelled
            after.append("planned")
            return self.answer

        with patch("downstream.run", side_effect=slow_run):
            request = ui._start("How does it work?", ask=True)
            request.cancel.set()
            release.set()
            with self.assertRaises(ui._Cancelled):
                request.future.result(timeout=5)
        self.assertEqual(after, [])
        self.assertEqual(request.events.get_nowait(), ("understanding", {}))


if __name__ == "__main__":
    unittest.main()
