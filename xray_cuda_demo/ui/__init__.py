"""Streamlit UI package (Section 12). Keeps app.py thin -- see
ui/services.py for the application-services layer that is the ONLY
place this package calls into the existing CPU/CUDA backend (Sections
1-11); no filter/kernel/benchmark logic is duplicated here.
"""
