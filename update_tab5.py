# Update Tab 5 script
with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

start_marker = """        # Safe defaults for ML Model info in municipal reports
        ckpt_cfg = spacenetra_model.CHECKPOINTS_CONFIG["fulltrain"]
        thresh_val = 0.20

        # ── NEW TAB: EXPERIMENTAL ML CHANGE DETECTION (S2LOOKING U-NET) ──
        with tab_ml_unet:"""

end_marker = """                            st.dataframe(eval_table, use_container_width=True, hide_index=True)
                            st.caption("Direct empirical evaluation comparing raw model output vs heuristic post-processed mask against the uploaded ground truth.")"""

print("start_marker in content:", start_marker in content)
print("end_marker in content:", end_marker in content)
