from manual_assistant.cleaning import (
    OutlineEntry,
    clean_pdf_page,
    pdf_pages_to_sections,
    wiki_html_to_sections,
)

OLD_STYLE_HTML = """
<div class="mw-parser-output">
  <div class="docnav">Previous | Next</div>
  <p>The FEM Workbench provides a finite element workflow.</p>
  <div id="toc" class="toc">Contents 1 Solvers</div>
  <h2><span class="mw-headline" id="Solvers">Solvers</span>
      <span class="mw-editsection">[edit]</span></h2>
  <p>CalculiX is the default solver.</p>
  <ul><li>Elmer</li><li>Z88</li></ul>
  <h3><span class="mw-headline" id="Installing_CalculiX">Installing CalculiX</span></h3>
  <p>It ships with FreeCAD.</p>
  <script>var x = 1;</script>
</div>
"""

NEW_STYLE_HTML = """
<div class="mw-content-ltr mw-parser-output">
  <div class="mw-heading mw-heading2"><h2 id="Usage">Usage</h2>
    <span class="mw-editsection">[edit]</span></div>
  <p>Select a face.</p>
</div>
"""


def test_wiki_sections_split_on_headings():
    sections = wiki_html_to_sections(OLD_STYLE_HTML, "FEM Workbench", "https://w/FEM_Workbench")
    assert [s.section for s in sections] == ["Introduction", "Solvers", "Installing CalculiX"]
    assert sections[0].url == "https://w/FEM_Workbench"
    assert sections[1].url == "https://w/FEM_Workbench#Solvers"
    assert "Elmer" in sections[1].text and "Z88" in sections[1].text


def test_wiki_chrome_is_removed():
    text = " ".join(s.text for s in wiki_html_to_sections(OLD_STYLE_HTML, "T", "u"))
    for junk in ("Previous", "Contents", "[edit]", "var x"):
        assert junk not in text


def test_new_mediawiki_heading_markup():
    sections = wiki_html_to_sections(NEW_STYLE_HTML, "Pad", "https://w/PartDesign_Pad")
    assert sections[0].section == "Usage"
    assert sections[0].url == "https://w/PartDesign_Pad#Usage"
    assert sections[0].text == "Select a face."


def test_pdf_sections_follow_outline():
    pages = [
        "Contents ... cover page",
        "1 Introduction\nCalculiX solves FE problems.\n2 *BOUNDARY\nKeyword for constraints.",
        "continued boundary text\n3 *CLOAD\nConcentrated loads.",
    ]
    outline = [
        OutlineEntry(("Introduction",), 1),
        OutlineEntry(("Input deck", "*BOUNDARY"), 1),
        OutlineEntry(("Input deck", "*CLOAD"), 2),
    ]
    sections = pdf_pages_to_sections(pages, outline, "CCX", "http://x/ccx.pdf")
    assert [s.section for s in sections] == [
        "Introduction",
        "Input deck > *BOUNDARY",
        "Input deck > *CLOAD",
    ]
    boundary = sections[1]
    assert boundary.url == "http://x/ccx.pdf#page=2"
    assert "Keyword for constraints" in boundary.text
    assert "continued boundary text" in boundary.text
    assert "Concentrated" not in boundary.text
    assert "cover page" not in " ".join(s.text for s in sections)


def test_pdf_title_not_found_starts_at_page_top():
    pages = ["intro", "body without heading"]
    outline = [OutlineEntry(("Intro",), 0), OutlineEntry(("Missing",), 1)]
    sections = pdf_pages_to_sections(pages, outline, "CCX", "u")
    assert sections[1].text == "body without heading"


def test_clean_pdf_page_removes_even_page_header():
    raw = "140\n\n6 THEORY\n\n6.4\n\nFluid Section Types: Gases\n\nBefore introducing"
    assert clean_pdf_page(raw, 140) == "6.4\n\nFluid Section Types: Gases\n\nBefore introducing"


def test_clean_pdf_page_removes_odd_page_header_and_late_page_number():
    raw = "6.4 Fluid Section Types: Gases\n\nM =\n\n141\n\nwhere κ is the speciﬁc heat"
    assert clean_pdf_page(raw, 141) == "M =\n\nwhere κ is the specific heat"


def test_clean_pdf_page_rejoins_hyphenated_words():
    assert clean_pdf_page("No cal-\n\nculation is done", 3) == "No calculation is done"
