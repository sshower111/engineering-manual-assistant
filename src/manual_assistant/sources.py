"""The public documents this project indexes."""

FREECAD_WIKI = "https://wiki.freecad.org"
FREECAD_API = f"{FREECAD_WIKI}/api.php"

# Curated English wiki pages, weighted towards FEM since that is where CalculiX fits in.
FREECAD_PAGES = [
    # FEM
    "FEM_Workbench",
    "FEM_tutorial",
    "FEM_CalculiX_Cantilever_3D",
    "FEM_Install",
    "FEM_Analysis",
    "FEM_MaterialSolid",
    "FEM_ConstraintFixed",
    "FEM_ConstraintForce",
    "FEM_ConstraintPressure",
    "FEM_ConstraintDisplacement",
    "FEM_ConstraintSelfWeight",
    "FEM_ConstraintTemperature",
    "FEM_MeshGmshFromShape",
    "FEM_MeshNetgenFromShape",
    "FEM_SolverCalculiXCcxTools",
    "FEM_SolverCalculiX",
    "FEM_ResultShow",
    "FEM_PostPipelineFromResult",
    "FEM_Preferences",
    # Part / PartDesign
    "Part_Workbench",
    "Part_Boolean",
    "Part_Extrude",
    "PartDesign_Workbench",
    "PartDesign_Body",
    "PartDesign_Pad",
    "PartDesign_Pocket",
    "PartDesign_Revolution",
    "PartDesign_Fillet",
    "PartDesign_Chamfer",
    "PartDesign_Hole",
    "PartDesign_LinearPattern",
    "PartDesign_PolarPattern",
    "PartDesign_Mirrored",
    # Sketcher
    "Sketcher_Workbench",
    "Sketcher_ConstrainCoincident",
    "Sketcher_ConstrainDistance",
    "Sketcher_ConstrainHorizontal",
    "Sketcher_ConstrainTangent",
    "Sketcher_ConstrainLock",
    "Sketcher_Dialog",
    # Other workbenches
    "TechDraw_Workbench",
    "TechDraw_View",
    "Draft_Workbench",
    "Mesh_Workbench",
    "Spreadsheet_Workbench",
    "Assembly_Workbench",
    # General concepts
    "Placement",
    "Units",
    "Expressions",
    "Topological_naming_problem",
    "Preferences_Editor",
    "Import_Export",
    "Std_Measure",
    "Macros",
    "Python_scripting_tutorial",
    "Installing_on_Windows",
    "Installing_on_Linux",
    "Installing_on_Mac",
    "Getting_started",
    "Feature_list",
]

# Newest first; the download script uses the first one that exists.
CALCULIX_PDF_CANDIDATES = [
    "http://www.dhondt.de/ccx_2.23.pdf",
    "http://www.dhondt.de/ccx_2.22.pdf",
]


def freecad_page_url(title: str) -> str:
    return f"{FREECAD_WIKI}/{title}"
