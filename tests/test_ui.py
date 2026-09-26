"""Streamlit interaction tests, with controlled backend results and no API calls."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from downstream import DownstreamError, DownstreamResult, EvidenceSource
from tests.test_downstream import passage, search_result

APP = str(Path(__file__).resolve().parents[1] / "ui" / "app.py")


class UITests(unittest.TestCase):
    def setUp(self):
        self.search = search_result()
        self.source = EvidenceSource(1, "2401.00001", "P2401.00001_0000", "Paper 2401.00001", 2024,
                                     "Method", "https://arxiv.org/abs/2401.00001v2", passage()["text"])
        self.answer = DownstreamResult("q-test", "How does it work?", "Low-rank updates [1].", "answered",
                                       [self.source], [1], self.search,
                                       {"total_ms": 100, "generation_ms": 20})

    def test_submit_reuses_search_and_shows_citation_and_original_text(self):
        with patch("downstream.run", return_value=self.answer) as answer, \
                patch("orchestration.run_query") as search, \
                patch("kb.store.load_passages", return_value={self.source.passage_id: passage()}):
            app = AppTest.from_file(APP).run()
            app.text_area[0].input("How does it work?")
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            answer.assert_called_once_with("How does it work?")
            search.assert_not_called()
            self.assertEqual([tab.label for tab in app.tabs], ["Paper QA", "Paper Search"])
            self.assertTrue(any("[1](#source-1)" in item.value for item in app.markdown))
            self.assertTrue(any(item.value == self.source.text for item in app.text))
            self.assertTrue(any("[1] Paper" in item.label for item in app.expander))
            self.assertEqual(len(app.get("download_button")), 1)
            app.run()
            answer.assert_called_once()

    def test_failed_followup_clears_stale_answer(self):
        with patch("downstream.run", side_effect=[self.answer, DownstreamError("Paper search failed.")]), \
                patch("kb.store.load_passages", return_value={}):
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
            example = app.selectbox[0].options[1]
            app.selectbox[0].select(example).run()
            self.assertEqual(app.text_area[0].value, example)
            answer.assert_not_called()
            search.assert_not_called()

    def test_search_only_does_not_generate_answer(self):
        with patch("downstream.run") as answer, patch("orchestration.run_query", return_value=self.search) as search, \
                patch("kb.store.load_passages", return_value={}):
            app = AppTest.from_file(APP).run()
            app.text_area[0].input("Find papers")
            app.button[1].click().run()
            self.assertEqual(len(app.exception), 0)
            search.assert_called_once_with("Find papers")
            answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
