from mi_bici.operations.load_report import write_summary


def test_summary_counts_only_valid_predictions(tmp_path):
    scenario = tmp_path / "escenario_10"
    scenario.mkdir()
    (tmp_path / "figuras").mkdir()  # No debe tratarse como escenario.

    (scenario / "locust_stats.csv").write_text(
        "Type,Name,Request Count,Failure Count,Requests/s,50%,95%,99%\n"
        "GET,/health,5,0,0.5,3,5,6\n"
        "POST,/predict,40,2,4.0,8,20,30\n"
        "POST,/predict [422 esperado],5,0,0.5,4,6,8\n"
        ",Aggregated,50,2,5.0,8,20,30\n",
        encoding="utf-8",
    )
    (scenario / "locust_stats_history.csv").write_text(
        "Timestamp,User Count,Type,Name,Requests/s,50%,95%,99%\n"
        "100,3,POST,/predict,2.0,8,15,20\n"
        "110,10,POST,/predict,4.0,8,20,30\n",
        encoding="utf-8",
    )

    rows = write_summary(tmp_path, tmp_path)

    assert len(rows) == 1
    assert rows[0]["escenario"] == "escenario_10"
    assert rows[0]["usuarios"] == 10
    assert rows[0]["duracion_s"] == 10
    assert rows[0]["peticiones"] == 40
    assert rows[0]["fallas"] == 2
    assert rows[0]["tasa_error"] == 0.05
    assert (tmp_path / "resumen.csv").is_file()
    assert (tmp_path / "resumen.md").is_file()
