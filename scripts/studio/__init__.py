"""Biologist Pipeline Studio — dependency-light logic for the Streamlit UI.

Pure Python + `requests` only (NO streamlit, NO torch) so it is unit-testable and
keeps the dashboard lightweight. `param_schema` maps biologist-friendly selections
to ClearML controller-task parameters; `clearml_launch` does the REST clone/enqueue.
"""
