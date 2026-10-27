# clean_tab5.py
with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

start_marker = """        # ── TAB 5: EXPERIMENTAL ML CHANGE DETECTION (S2LOOKING U-NET) ──
        with tab_ml_unet:"""

end_marker = """                            st.dataframe(ev_cmp_df, use_container_width=True, hide_index=True)"""

print("start_marker in content:", start_marker in content)
print("end_marker in content:", end_marker in content)
