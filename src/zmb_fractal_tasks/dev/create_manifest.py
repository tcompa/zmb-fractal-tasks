"""
Generate JSON schemas for task arguments afresh, and write them
to the package manifest.
"""
from fractal_tasks_core.dev.create_manifest import create_manifest

if __name__ == "__main__":
    PACKAGE = "zmb_fractal_tasks"
    AUTHORS = "Flurin Sturzenegger"
    docs_link = ""
    if docs_link:
        create_manifest(package=PACKAGE, authors=AUTHORS, docs_link=docs_link)
    else:
        create_manifest(package=PACKAGE, authors=AUTHORS)
