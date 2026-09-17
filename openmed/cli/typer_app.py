"""Draft Typer-powered CLI for OpenMed.

This supplements the existing argparse CLI without breaking it. It is
optional: install extras `pip install .[cli]` or add Typer/Rich to your
environment, then run:

    python -m openmed.cli.typer_app --help
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

try:  # soft dependency to avoid import errors in base installs
    import typer
    from rich import print as rprint
    from rich.console import Console
    from rich.table import Table
except ImportError:  # pragma: no cover - optional surface
    typer = None
    Console = None
    Table = None
    rprint = print

from openmed import analyze_text, get_model_max_length, list_models
from openmed.ambient import (
    AmbientRedactionPipeline,
    MicrophoneAudioSource,
    WavFileAudioSource,
    anonymize_wav_file,
)
from openmed.cli.main import _format_models_size_table, build_models_size_report
from openmed.core.capabilities import backend_status
from openmed.core.config import (
    OpenMedConfig,
    get_config,
    load_config_from_file,
    resolve_config_path,
    save_config_to_file,
    set_config,
)
from openmed.core.model_integrity import (
    ModelIntegrityError,
    verify_cached_models,
)
from openmed.graph import EntityGraph, GraphRAGRetriever, build_entity_graph
from openmed.ner import (
    NerRequest,
    build_index,
    ensure_gliner2_available,
    ensure_gliner_available,
    write_index,
)
from openmed.ner import (
    infer as zs_infer,
)


def _ensure_typer():
    if typer is None:
        raise RuntimeError(
            "Typer/Rich not installed. Install with `pip install .[cli]` or "
            "`pip install typer rich`."
        )


def _load_config(config_path: Optional[Path]) -> OpenMedConfig:
    if config_path:
        try:
            cfg = load_config_from_file(config_path)
            set_config(cfg)
            return cfg
        except FileNotFoundError:
            pass
    return get_config()


def _echo_json(payload: object) -> None:
    rprint(json.dumps(payload, indent=2, ensure_ascii=False))


def _render_table(title: str, headers: List[str], rows: List[List[str]]) -> None:
    if Console is None or Table is None:
        for row in rows:
            rprint(" | ".join(row))
        return
    table = Table(title=title)
    for head in headers:
        table.add_column(head)
    for row in rows:
        table.add_row(*[str(cell) for cell in row])
    Console().print(table)


def build_app():
    """Build and return the Typer application."""
    _ensure_typer()

    app = typer.Typer(help="OpenMed Typer CLI (draft).")
    models_app = typer.Typer(help="Model discovery commands.")
    cli_app = typer.Typer(help="Config utilities.")
    zero_app = typer.Typer(help="Zero-shot (GLiNER/GLiNER2) utilities.")
    ambient_app = typer.Typer(help="Real-time ambient speech redaction.")
    graph_app = typer.Typer(help="Local entity co-occurrence graph and GraphRAG retrieval.")

    # ------------------------------------------------------------------
    # analyze
    # ------------------------------------------------------------------
    @app.command("analyze")
    def analyze(
        text: Optional[str] = typer.Option(
            None, "--text", "-t", help="Inline text to analyse."
        ),
        input_file: Optional[Path] = typer.Option(
            None, "--input-file", "-f", help="Path to a text file."
        ),
        model: str = typer.Option(
            "disease_detection_superclinical",
            "--model",
            "-m",
            help="Registry key or HF model id.",
        ),
        output_format: str = typer.Option(
            "dict", "--format", "-o", help="dict|json|html|csv"
        ),
        confidence_threshold: Optional[float] = typer.Option(
            None, "--threshold", "-c", help="Minimum confidence to keep entities."
        ),
        group_entities: bool = typer.Option(
            False, "--group", help="Merge adjacent spans of the same label."
        ),
        no_confidence: bool = typer.Option(
            False, "--no-confidence", help="Exclude confidence values."
        ),
        sentence_detection: bool = typer.Option(
            True,
            "--sentence-detection/--no-sentence-detection",
            help="Toggle sentence splitting.",
        ),
        config_path: Optional[Path] = typer.Option(
            None, "--config-path", help="Override config path."
        ),
    ):
        """Analyse text with an OpenMed model and pretty-print the result."""
        cfg = _load_config(config_path)
        if text is None and input_file is None:
            raise typer.BadParameter("Provide --text or --input-file.")
        payload = text
        if input_file:
            payload = input_file.read_text(encoding="utf-8")

        result = analyze_text(
            payload,
            model_name=model,
            output_format=output_format,
            confidence_threshold=confidence_threshold,
            group_entities=group_entities,
            include_confidence=not no_confidence,
            sentence_detection=sentence_detection,
            config=cfg,
        )

        if hasattr(result, "to_dict"):
            _echo_json(result.to_dict())
        else:
            rprint(result)

    # ------------------------------------------------------------------
    # models
    # ------------------------------------------------------------------
    @models_app.command("list")
    def list_available_models(
        include_remote: bool = typer.Option(
            False, "--include-remote", help="Query Hugging Face Hub."
        ),
        config_path: Optional[Path] = typer.Option(
            None, "--config-path", help="Override config path."
        ),
    ):
        cfg = _load_config(config_path)
        models = list_models(
            include_registry=True, include_remote=include_remote, config=cfg
        )
        rows = [[m] for m in models]
        _render_table("Models", ["model_id"], rows)

    @models_app.command("info")
    def model_info(
        model_key: str = typer.Argument(..., help="Registry key or HF id."),
        config_path: Optional[Path] = typer.Option(
            None, "--config-path", help="Override config path."
        ),
    ):
        cfg = _load_config(config_path)
        max_len = get_model_max_length(model_key, config=cfg)
        _echo_json({"model_key": model_key, "max_length": max_len})

    @models_app.command("verify")
    def models_verify(
        model_id: Optional[str] = typer.Argument(
            None,
            help="Registry model id or local model directory.",
        ),
        all_models: bool = typer.Option(
            False,
            "--all",
            help="Verify every cached model with integrity metadata.",
        ),
        config_path: Optional[Path] = typer.Option(
            None,
            "--config-path",
            help="Override config path.",
        ),
    ):
        """Verify cached model artifacts without network access."""
        if (model_id is None) == (not all_models):
            raise typer.BadParameter("Provide MODEL_ID or --all, but not both.")
        cfg = _load_config(config_path)
        try:
            results = verify_cached_models(
                cache_dir=str(cfg.cache_dir),
                model_id=None if all_models else model_id,
            )
        except ModelIntegrityError as exc:
            _render_table(
                "Model integrity",
                ["model_id", "status", "expected", "actual", "files"],
                [
                    [
                        exc.model_id,
                        "FAIL",
                        exc.expected_sha256,
                        exc.actual_sha256,
                        "-",
                    ]
                ],
            )
            rprint(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        except (OSError, ValueError) as exc:
            rprint(f"[red]Model integrity verification failed: {exc}[/red]")
            raise typer.Exit(code=1) from exc

        rows = [
            [
                result.model_id,
                "PASS",
                result.expected_sha256,
                result.actual_sha256,
                str(result.files_checked),
            ]
            for result in results
        ]
        _render_table(
            "Model integrity",
            ["model_id", "status", "expected", "actual", "files"],
            rows,
        )
        if not rows:
            rprint("No verified model caches found.")

    @models_app.command("size")
    def model_size(
        model_key: Optional[str] = typer.Argument(
            None, help="Optional registry alias or full model repository id."
        ),
        remote: bool = typer.Option(
            False,
            "--remote",
            help="Refine snapshot sizes from Hugging Face Hub metadata.",
        ),
        budget_mb: Optional[float] = typer.Option(
            None,
            "--budget-mb",
            min=0,
            help="Only show models needing at most this many MB to download.",
        ),
        output_format: str = typer.Option(
            "table",
            "--format",
            help="Output format: table or json.",
        ),
    ):
        """Show offline-safe download, disk, and peak RAM estimates."""

        if output_format not in {"table", "json"}:
            raise typer.BadParameter("--format must be 'table' or 'json'")
        try:
            report = build_models_size_report(
                model_key,
                budget_mb=budget_mb,
                remote=remote,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            typer.echo(f"Failed to inspect model sizes: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        if not report["models"]:
            if budget_mb is None:
                message = "No model size metadata is available."
            else:
                message = f"No models fit the {budget_mb:g} MB download budget."
            typer.echo(message, err=True)
            raise typer.Exit(code=1)

        for warning in report["warnings"]:
            typer.echo(f"Remote size lookup warning: {warning}", err=True)
        if output_format == "json":
            typer.echo(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            typer.echo(_format_models_size_table(report), nl=False)

    app.add_typer(models_app, name="models")

    # ------------------------------------------------------------------
    # config
    # ------------------------------------------------------------------
    @cli_app.command("show")
    def config_show(config_path: Optional[Path] = typer.Option(None, "--config-path")):
        cfg_path = resolve_config_path(config_path)
        try:
            cfg = load_config_from_file(cfg_path)
            source = str(cfg_path)
        except FileNotFoundError:
            cfg = get_config()
            source = "defaults (not yet saved)"
        payload = cfg.to_dict()
        payload["_source"] = source
        _echo_json(payload)

    @cli_app.command("set")
    def config_set(
        key: str,
        value: Optional[str] = typer.Argument(None),
        unset: bool = typer.Option(False, "--unset", help="Clear the value."),
        config_path: Optional[Path] = typer.Option(None, "--config-path"),
    ):
        cfg_path = resolve_config_path(config_path)
        try:
            cfg = load_config_from_file(cfg_path)
        except FileNotFoundError:
            cfg = get_config()
        cfg_dict = cfg.to_dict()
        if key not in cfg_dict:
            raise typer.BadParameter(
                f"Unknown key '{key}'. Valid: {', '.join(cfg_dict.keys())}"
            )
        cfg_dict[key] = None if unset else value
        new_cfg = OpenMedConfig.from_dict(cfg_dict)
        set_config(new_cfg)
        save_config_to_file(new_cfg, cfg_path)
        rprint(f"[green]Updated {key} -> {cfg_dict[key]} in {cfg_path}[/green]")

    app.add_typer(cli_app, name="config")

    # ------------------------------------------------------------------
    # zero-shot (GLiNER/GLiNER2)
    # ------------------------------------------------------------------
    @zero_app.command("deps")
    def zero_deps():
        messages = []
        try:
            ensure_gliner_available()
            messages.append("GLiNER v1: ok")
        except Exception as exc:  # pragma: no cover
            messages.append(f"GLiNER v1: missing ({exc})")
        try:
            ensure_gliner2_available()
            messages.append("GLiNER v2: ok")
        except Exception as exc:  # pragma: no cover
            messages.append(f"GLiNER v2: missing ({exc})")
        for line in messages:
            rprint(line)

    @zero_app.command("index")
    def zero_index(
        models_dir: Path = typer.Argument(
            ..., help="Root directory containing zero-shot models."
        ),
        output: Optional[Path] = typer.Option(
            None, "--output", "-o", help="Path to write index.json"
        ),
        pretty: bool = typer.Option(
            True, "--pretty/--compact", help="Pretty-print JSON."
        ),
    ):
        index = build_index(models_dir)
        out_path = output or (models_dir / "index.json")
        write_index(index, out_path, pretty=pretty)
        rprint(f"[green]Index written to {out_path}[/green]")

    @zero_app.command("infer")
    def zero_infer(
        text: str = typer.Argument(..., help="Input text for zero-shot NER."),
        model_id: str = typer.Option(
            ..., "--model-id", "-m", help="Model id from index."
        ),
        labels: Optional[str] = typer.Option(
            None, "--labels", "-l", help="Comma-separated label list (optional)."
        ),
        domain: Optional[str] = typer.Option(
            None, "--domain", "-d", help="Domain hint."
        ),
        threshold: float = typer.Option(
            0.5, "--threshold", "-c", help="Score threshold."
        ),
        index_path: Optional[Path] = typer.Option(
            None,
            "--index-path",
            "-i",
            help="Path to index.json (defaults to models/index.json).",
        ),
    ):
        label_list = [label.strip() for label in labels.split(",")] if labels else None
        request = NerRequest(
            model_id=model_id,
            text=text,
            labels=label_list,
            domain=domain,
            threshold=threshold,
        )
        response = zs_infer(request, index_path=index_path)
        _echo_json(response.to_dict())

    app.add_typer(zero_app, name="zero")

    # ------------------------------------------------------------------
    # ambient (real-time speech redaction)
    # ------------------------------------------------------------------
    @ambient_app.command("deps")
    def ambient_deps():
        for name in ("speech", "mic", "voice_privacy"):
            status = backend_status(name)
            state = "ok" if status.available else f"missing ({status.install_hint})"
            rprint(f"{name}: {state}")

    @ambient_app.command("anonymize-voice")
    def ambient_anonymize_voice(
        input_path: Path = typer.Argument(..., help="16-bit PCM .wav file to anonymize."),
        output_path: Path = typer.Argument(..., help="Where to write the anonymized .wav."),
        mcadams: float = typer.Option(
            0.8,
            "--mcadams",
            help="McAdams coefficient in (0, 1.5]; further from 1.0 warps the voice more.",
        ),
    ):
        """Warp a recorded file's voiceprint without changing what was said."""
        anonymize_wav_file(input_path, output_path, mcadams_coefficient=mcadams)
        rprint(f"[green]Anonymized voice written to {output_path}[/green]")

    @ambient_app.command("file")
    def ambient_file(
        audio_path: Path = typer.Argument(..., help="16-bit PCM .wav file to replay."),
        model_size: str = typer.Option(
            "small.en", "--model", "-m", help="faster-whisper model size."
        ),
        deid_method: str = typer.Option(
            "mask", "--deid-method", help="mask|replace|hash|remove|shift_dates."
        ),
        chunk_seconds: float = typer.Option(
            3.0, "--chunk-seconds", help="Seconds of audio per transcription chunk."
        ),
        language: Optional[str] = typer.Option(
            None, "--language", help="Force a transcription language (e.g. 'en')."
        ),
    ):
        """Replay a recorded encounter and print de-identified transcript segments."""
        source = WavFileAudioSource(audio_path, chunk_seconds=chunk_seconds)
        pipeline = AmbientRedactionPipeline(
            model_size=model_size,
            deid_method=deid_method,  # type: ignore[arg-type]
            language=language,
        )
        for redacted in pipeline.stream(source):
            _echo_json(redacted.to_dict())

    @ambient_app.command("mic")
    def ambient_mic(
        model_size: str = typer.Option(
            "small.en", "--model", "-m", help="faster-whisper model size."
        ),
        deid_method: str = typer.Option(
            "mask", "--deid-method", help="mask|replace|hash|remove|shift_dates."
        ),
        chunk_seconds: float = typer.Option(
            3.0, "--chunk-seconds", help="Seconds of audio per transcription chunk."
        ),
        max_duration_seconds: Optional[float] = typer.Option(
            None, "--max-duration", help="Stop capture after this many seconds."
        ),
        language: Optional[str] = typer.Option(
            None, "--language", help="Force a transcription language (e.g. 'en')."
        ),
    ):
        """Redact live microphone speech in real time, one chunk at a time."""
        source = MicrophoneAudioSource(
            chunk_seconds=chunk_seconds,
            max_duration_seconds=max_duration_seconds,
        )
        pipeline = AmbientRedactionPipeline(
            model_size=model_size,
            deid_method=deid_method,  # type: ignore[arg-type]
            language=language,
        )
        rprint("[green]Listening... press Ctrl+C to stop.[/green]")
        for redacted in pipeline.stream(source):
            _echo_json(redacted.to_dict())

    app.add_typer(ambient_app, name="ambient")

    # ------------------------------------------------------------------
    # graph (local entity co-occurrence graph / GraphRAG)
    # ------------------------------------------------------------------
    @graph_app.command("build")
    def graph_build(
        input_files: List[Path] = typer.Argument(
            ..., help="Text files to build the graph from (one document each)."
        ),
        output: Path = typer.Option(
            ..., "--output", "-o", help="Where to write the graph as JSON."
        ),
        models: str = typer.Option(
            "disease_detection_superclinical",
            "--models",
            help="Comma-separated OpenMed NER model names to run over every document.",
        ),
        window: str = typer.Option(
            "sentence", "--window", help="Co-occurrence window: sentence|document."
        ),
        confidence_threshold: float = typer.Option(
            0.5, "--threshold", help="Minimum entity confidence to include."
        ),
    ):
        """Build a local entity co-occurrence graph from a set of text files."""
        documents = {
            path.stem: path.read_text(encoding="utf-8") for path in input_files
        }
        model_names = [name.strip() for name in models.split(",") if name.strip()]
        graph = build_entity_graph(
            documents,
            model_names=model_names,
            cooccurrence_window=window,  # type: ignore[arg-type]
            confidence_threshold=confidence_threshold,
        )
        output.write_text(json.dumps(graph.to_dict(), indent=2), encoding="utf-8")
        rprint(
            f"[green]Graph written to {output}[/green] "
            f"({len(graph.nodes)} nodes, {len(graph.edges)} edges)"
        )

    @graph_app.command("query")
    def graph_query(
        question: str = typer.Argument(..., help="Free-text question to ground."),
        graph_path: Path = typer.Option(
            ..., "--graph", "-g", help="Path to a graph.json from `openmed graph build`."
        ),
        models: str = typer.Option(
            "disease_detection_superclinical",
            "--models",
            help="Comma-separated OpenMed NER model names to extract query entities.",
        ),
        top_k: int = typer.Option(
            5, "--top-k", help="Max neighbors to return per matched entity."
        ),
    ):
        """Return the graph neighborhood of every entity recognized in a question."""
        data = json.loads(graph_path.read_text(encoding="utf-8"))
        graph = EntityGraph.from_dict(data)
        model_names = [name.strip() for name in models.split(",") if name.strip()]
        retriever = GraphRAGRetriever(graph, model_names=model_names, top_k_per_entity=top_k)
        context = retriever.retrieve(question)
        _echo_json(context.to_dict())

    app.add_typer(graph_app, name="graph")

    return app


def main() -> None:
    """Run the Typer command-line application."""
    build_app()()


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except RuntimeError as exc:
        rprint(f"[red]{exc}[/red]")
