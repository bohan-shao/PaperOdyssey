# Fonts

Latin Modern Roman 10 (version 2.007), the default face of LaTeX documents, used by the UI for
its title, answer, references and excerpts. Regular, italic, bold and bold italic, unmodified
WOFF2 files as distributed with LaTeX.css (https://github.com/vincentdoerig/latex-css, `fonts/`).

Copyright: the Polish TeX Users Group GUST, on behalf of the Local TeX Users Groups.
Licence: GUST Font Licence, legally identical to the LaTeX Project Public License 1.3c; see
`GUST-FONT-LICENSE.txt`, or https://www.gust.org.pl/fonts/licenses/GUST-FONT-LICENSE.txt.

Streamlit serves this folder at `app/static/fonts/` (`server.enableStaticServing` in
`.streamlit/config.toml`), so the faces load from the app itself, also offline.
