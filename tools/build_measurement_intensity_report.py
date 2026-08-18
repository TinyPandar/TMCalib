"""Build the canonical Data Analytics artifact for the intensity diagnosis."""

import argparse
import json
import os
from datetime import datetime


def _source(source_id, label, path, description, files):
    return (
        {
            "id": source_id,
            "label": label,
            "path": path,
        },
        {
            "id": source_id,
            "query": {
                "engine": "python",
                "language": "python",
                "description": description,
                "tables_used": files,
                "executed_at": datetime.now().astimezone().isoformat(),
            },
        },
    )


def build_artifact(diagnostic):
    profile = diagnostic["current_profile"]
    focus = diagnostic["focus_diagnostics"]
    generated_at = diagnostic["generated_at"]

    localization_bands = [
        {
            "distance_band": "0-1 px",
            "target_fraction": focus["within_1px_fraction"],
            "lower_px": 0,
            "upper_px": 1,
        },
        {
            "distance_band": ">1-2 px",
            "target_fraction": (
                focus["within_2px_fraction"] - focus["within_1px_fraction"]
            ),
            "lower_px": 1,
            "upper_px": 2,
        },
        {
            "distance_band": ">2-5 px",
            "target_fraction": (
                focus["within_5px_fraction"] - focus["within_2px_fraction"]
            ),
            "lower_px": 2,
            "upper_px": 5,
        },
        {
            "distance_band": ">5-20 px",
            "target_fraction": (
                1.0
                - focus["over_20px_fraction"]
                - focus["within_5px_fraction"]
            ),
            "lower_px": 5,
            "upper_px": 20,
        },
        {
            "distance_band": ">20-50 px",
            "target_fraction": (
                focus["over_20px_fraction"] - focus["over_50px_fraction"]
            ),
            "lower_px": 20,
            "upper_px": 50,
        },
        {
            "distance_band": ">50-100 px",
            "target_fraction": (
                focus["over_50px_fraction"] - focus["over_100px_fraction"]
            ),
            "lower_px": 50,
            "upper_px": 100,
        },
        {
            "distance_band": ">100 px",
            "target_fraction": focus["over_100px_fraction"],
            "lower_px": 100,
            "upper_px": 181,
        },
    ]

    headline = [
        {
            "native_median_dn": profile["native_8bit_quantiles"]["0.5"],
            "fraction_le_2_dn": profile[
                "at_or_below_2_native_codes_fraction"
            ],
            "zero_fraction": profile["zero_fraction"],
            "within_2px_fraction": focus["within_2px_fraction"],
            "over_20px_fraction": focus["over_20px_fraction"],
            "intensity_pbr_pearson": focus[
                "measurement_mean_vs_pbr_pearson"
            ],
        }
    ]

    quality_checks = [
        {
            "check": "Native sensor occupancy",
            "status": "High concern",
            "evidence": "Median 2 DN; 52.6% at or below 2 DN",
            "impact": "Quantization and dark/read noise can perturb recovered amplitude",
        },
        {
            "check": "Zero-valued samples",
            "status": "High concern",
            "evidence": "7.95% of 1.07B samples are zero",
            "impact": "sqrt(intensity) collapses these samples to zero amplitude",
        },
        {
            "check": "Saturation",
            "status": "Pass",
            "evidence": "0 samples at native 250-255 DN; stored max 1172",
            "impact": "There is exposure headroom before clipping",
        },
        {
            "check": "Batch stability",
            "status": "Pass with monitoring",
            "evidence": "Batch means 54.98-56.55; first-to-last drift +2.10%",
            "impact": "Temporal drift is too small to explain the localization split",
        },
        {
            "check": "Frame integrity",
            "status": "Pass",
            "evidence": "0 blank frames; 0 exact duplicate groups",
            "impact": "No evidence of gross acquisition loss or buffer replay",
        },
        {
            "check": "Focus localization",
            "status": "High concern",
            "evidence": "55.7% within 2 px; 33.9% miss by more than 20 px",
            "impact": "Points to mapping/registration/encoding issues beyond uniform SNR",
        },
    ]

    manifest_sources = []
    canonical_sources = []
    source_definitions = [
        _source(
            "measurement_current",
            "2026-08-06 active512 measurement",
            "measurements_128_px4_active512_full_memmap.npy",
            "Full scan of all 65,536 frames and 128x128 output pixels.",
            ["measurements_128_px4_active512_full_memmap.npy"],
        ),
        _source(
            "focus_current",
            "2026-08-06 active512 pixel-wise focus",
            (
                "pixelwise_focus_results_128_px4_active512/"
                "pixelwise_focus_128_px4_active512_20260806_161522_"
                "stride1_batch1000_n16384_points.csv"
            ),
            "All 16,384 requested focus positions and their PBR/localization results.",
            [
                "pixelwise_focus_128_px4_active512_20260806_161522_"
                "stride1_batch1000_n16384_points.csv"
            ],
        ),
        _source(
            "diagnostic_aggregate",
            "Reproducible intensity diagnostic",
            "measurement_intensity_diagnostic_20260806.json",
            "Joined measurement, recovery, and pixel-wise focus aggregates produced by the companion diagnostic script.",
            [
                "measurement_intensity_diagnostic_20260806.json",
                "diagnose_measurement_intensity_128.py",
            ],
        ),
        _source(
            "camera_pipeline",
            "128 calibration acquisition code",
            "calibrate_128x128.py",
            "Camera encoding, exposure, gain, DMD timing, recovery, and focus definitions used by the application.",
            ["calibrate_128x128.py"],
        ),
    ]
    for manifest_source, canonical_source in source_definitions:
        manifest_sources.append(manifest_source)
        canonical_sources.append(canonical_source)

    charts = [
        {
            "id": "native_histogram",
            "title": "Native 8-bit intensity distribution",
            "subtitle": "All 1.07B camera samples; 52.6% are at or below 2 DN",
            "type": "bar",
            "intent": "comparison",
            "question": "How much of the measurement occupies only the lowest native camera codes?",
            "rationale": "Pre-binned bars make the strong concentration in the first few discrete 8-bit codes directly visible.",
            "dataset": "native_histogram_bins",
            "sourceId": "measurement_current",
            "valueFormat": "percent",
            "palette": {"kind": "sequential", "name": "blue"},
            "labels": {"values": "auto"},
            "encodings": {
                "x": {
                    "field": "native_code_bin",
                    "type": "nominal",
                    "label": "Native 8-bit code bin",
                },
                "y": {
                    "field": "sample_fraction",
                    "type": "quantitative",
                    "label": "Share of samples",
                    "format": "percent",
                },
                "tooltip": [
                    {
                        "field": "sample_count",
                        "type": "quantitative",
                        "label": "Samples",
                        "format": "number",
                    }
                ],
            },
        },
        {
            "id": "localization_bands",
            "title": "Focus localization error distribution",
            "subtitle": "Brightest-peak distance from each requested target across 16,384 points",
            "type": "bar",
            "intent": "comparison",
            "question": "Is the weak recovery broad and uniform or split between localized successes and large misses?",
            "rationale": "Distance bands reveal the mixed success/failure pattern that an average PBR alone hides.",
            "dataset": "localization_bands",
            "sourceId": "focus_current",
            "valueFormat": "percent",
            "palette": {"kind": "sequential", "name": "orange"},
            "labels": {"values": "auto"},
            "encodings": {
                "x": {
                    "field": "distance_band",
                    "type": "nominal",
                    "label": "Peak distance from target",
                },
                "y": {
                    "field": "target_fraction",
                    "type": "quantitative",
                    "label": "Share of requested targets",
                    "format": "percent",
                },
                "tooltip": [
                    {
                        "field": "upper_px",
                        "type": "quantitative",
                        "label": "Upper bound (px)",
                        "format": "number",
                    }
                ],
            },
        },
        {
            "id": "pbr_by_intensity_quintile",
            "title": "Mean PBR by measurement-intensity quintile",
            "subtitle": "Current active512 run; Q1 is the dimmest output-pixel quintile",
            "type": "bar",
            "intent": "comparison",
            "question": "Do dimmer output pixels systematically recover with lower PBR?",
            "rationale": "Five ordered output-intensity groups provide a direct monotonicity check without overfitting individual pixels.",
            "dataset": "intensity_quintiles",
            "sourceId": "diagnostic_aggregate",
            "valueFormat": "number",
            "palette": {"kind": "sequential", "name": "blue"},
            "labels": {"values": "all"},
            "encodings": {
                "x": {
                    "field": "intensity_quintile",
                    "type": "ordinal",
                    "label": "Measurement-intensity quintile",
                },
                "y": {
                    "field": "pbr_mean",
                    "type": "quantitative",
                    "label": "Mean target PBR",
                },
                "tooltip": [
                    {
                        "field": "within_2px_fraction",
                        "type": "quantitative",
                        "label": "Within 2 px",
                        "format": "percent",
                    },
                    {
                        "field": "measurement_mean_min",
                        "type": "quantitative",
                        "label": "Min mean intensity",
                        "format": "number",
                    },
                    {
                        "field": "measurement_mean_max",
                        "type": "quantitative",
                        "label": "Max mean intensity",
                        "format": "number",
                    },
                ],
            },
        },
        {
            "id": "batch_mean",
            "title": "Measurement mean by acquisition batch",
            "subtitle": "66 batches; stored intensity stays within 54.98-56.55 DN",
            "type": "line",
            "intent": "trend",
            "question": "Did temporal intensity drift during the full measurement explain the recovery pattern?",
            "rationale": "The ordered 66-batch series is long enough to reveal acquisition drift and discontinuities.",
            "dataset": "batch_profile",
            "sourceId": "measurement_current",
            "valueFormat": "number",
            "palette": {"kind": "sequential", "name": "blue"},
            "labels": {"values": "endpoints"},
            "encodings": {
                "x": {
                    "field": "batch",
                    "type": "ordinal",
                    "label": "Acquisition batch",
                },
                "y": {
                    "field": "mean_intensity",
                    "type": "quantitative",
                    "label": "Mean stored intensity",
                },
                "tooltip": [
                    {
                        "field": "zero_fraction",
                        "type": "quantitative",
                        "label": "Zero fraction",
                        "format": "percent",
                    }
                ],
            },
        },
    ]

    cards = [
        {
            "id": "native_median_card",
            "description": "Median of the true Polarized8 camera codes before 12-bit range scaling.",
            "dataset": "headline",
            "sourceId": "measurement_current",
            "metrics": [
                {
                    "label": "Median native DN",
                    "field": "native_median_dn",
                    "format": "number",
                }
            ],
        },
        {
            "id": "low_code_card",
            "description": "Share of all probe-pixel samples occupying native codes 0, 1, or 2.",
            "dataset": "headline",
            "sourceId": "measurement_current",
            "metrics": [
                {
                    "label": "At or below 2 DN",
                    "field": "fraction_le_2_dn",
                    "format": "percent",
                }
            ],
        },
        {
            "id": "within_two_card",
            "description": "Requested positions whose brightest observed peak is no more than 2 pixels away.",
            "dataset": "headline",
            "sourceId": "focus_current",
            "metrics": [
                {
                    "label": "Localized within 2 px",
                    "field": "within_2px_fraction",
                    "format": "percent",
                }
            ],
        },
        {
            "id": "large_miss_card",
            "description": "Requested positions whose brightest observed peak is more than 20 pixels away.",
            "dataset": "headline",
            "sourceId": "focus_current",
            "metrics": [
                {
                    "label": "Missed by over 20 px",
                    "field": "over_20px_fraction",
                    "format": "percent",
                }
            ],
        },
        {
            "id": "correlation_card",
            "description": "Pearson correlation across output pixels between calibration mean intensity and target PBR.",
            "dataset": "headline",
            "sourceId": "diagnostic_aggregate",
            "metrics": [
                {
                    "label": "Intensity-PBR correlation",
                    "field": "intensity_pbr_pearson",
                    "format": "number",
                }
            ],
        },
    ]

    tables = [
        {
            "id": "run_comparison",
            "title": "Measurement and focus outcomes across preserved runs",
            "subtitle": "The two active512 runs are the closest comparison; the legacy mapping is contextual",
            "dataset": "run_comparison",
            "sourceId": "diagnostic_aggregate",
            "defaultSort": {"field": "run", "direction": "asc"},
            "columns": [
                {"field": "run", "label": "Run", "type": "text"},
                {"field": "mapping", "label": "Mapping", "type": "text"},
                {
                    "field": "measurement_mean",
                    "label": "Measurement mean",
                    "format": "number",
                },
                {
                    "field": "zero_fraction",
                    "label": "Zero share",
                    "format": "percent",
                },
                {
                    "field": "pbr_mean",
                    "label": "Mean PBR",
                    "format": "number",
                },
                {
                    "field": "peak_distance_median_px",
                    "label": "Median peak distance (px)",
                    "format": "number",
                },
            ],
        },
        {
            "id": "quality_checks",
            "title": "Data-quality and recovery checks",
            "subtitle": "Severity reflects risk to phase recovery and target localization",
            "dataset": "quality_checks",
            "sourceId": "diagnostic_aggregate",
            "defaultSort": {"field": "check", "direction": "asc"},
            "columns": [
                {"field": "check", "label": "Check", "type": "text"},
                {"field": "status", "label": "Assessment", "type": "text"},
                {"field": "evidence", "label": "Evidence", "type": "text"},
                {"field": "impact", "label": "Why it matters", "type": "text"},
            ],
        },
    ]

    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# 128x128 measurement intensity and recovery diagnosis",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## Technical summary\n\n"
                "**Low light is real, but the available evidence does not support it as the primary cause of the current recovery failures.** "
                "The latest measurement has a native 8-bit median of only 2 DN; 52.6% of all samples are at or below 2 DN and 7.95% are zero. "
                "That makes quantization and unmeasured dark/read noise a meaningful secondary limitation. However, raising active512 mean intensity roughly fourfold between 5 and 6 August did not improve mean PBR (14.20 to 13.64), and current per-output measurement intensity is essentially unrelated to PBR (Pearson r=0.008).\n\n"
                "The stronger failure signature is localization: 55.7% of requested targets place the brightest peak within 2 px, while 33.9% miss by more than 20 px. "
                "That split is more consistent with coordinate mapping, registration, TM-row indexing, or hologram encoding errors than with one uniform SNR ceiling. Confidence is high that low light is not the sole or leading explanation; confidence is moderate on the exact alternative until a controlled exposure sweep and coordinate impulse test are run."
            ),
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [
                "native_median_card",
                "low_code_card",
                "within_two_card",
                "large_miss_card",
                "correlation_card",
            ],
        },
        {
            "id": "low_occupancy_finding",
            "type": "markdown",
            "sourceId": "measurement_current",
            "body": (
                "## Sensor occupancy is low enough to hurt precision\n\n"
                "The camera is configured as Polarized8 and then numerically stretched into a 12-bit uint16 range. The stored median of 32 therefore represents only 2 native camera codes, not 12-bit precision. "
                "More than half of the full 1.07B-sample tensor occupies native codes 0-2, while no samples approach native saturation. This creates real sensitivity to quantization and dark/read noise, especially because recovery uses `sqrt(intensity)` with dark level set to zero.\n\n"
                "**Implication:** exposure or native bit depth should be improved, but this alone should not be expected to fix the spatially large focus misses."
            ),
        },
        {"id": "native_histogram_block", "type": "chart", "chartId": "native_histogram"},
        {
            "id": "comparison_finding",
            "type": "markdown",
            "sourceId": "diagnostic_aggregate",
            "body": (
                "## More light did not produce better active512 recovery\n\n"
                "The closest preserved comparison is between the two active512 runs. Mean stored intensity rose from 13.96 to 55.85 (4.0x) and the zero share fell from 29.5% to 7.95%, yet mean PBR moved slightly down from 14.20 to 13.64. "
                "The older legacy-mapping run was brighter still (mean 74.4) but had mean PBR 3.18 and a median peak error of 46.9 px. Because its mapping differs, it is not a controlled exposure test; it nevertheless shows that high intensity is not sufficient for good recovery.\n\n"
                "The final relative amplitude residual is also similar for the brighter legacy run and the current run (0.212 versus 0.218), so solver fit alone does not explain the large difference in focus localization."
            ),
        },
        {"id": "run_comparison_block", "type": "table", "tableId": "run_comparison"},
        {
            "id": "localization_finding",
            "type": "markdown",
            "sourceId": "focus_current",
            "body": (
                "## The dominant failure mode is a split between good localization and large misses\n\n"
                "A mean target PBR of 13.64 looks acceptable in isolation, but the spatial check is mixed: 55.7% of targets land within 2 px and 64.0% within 5 px, while 33.9% miss by more than 20 px and 24.5% miss by more than 50 px. "
                "A uniform low-SNR limit would more naturally produce broad degradation; this bimodal-like pattern instead prioritizes coordinate transforms, axis flips, row indexing, active-region offsets, and encoding consistency.\n\n"
                "**Implication:** use peak-distance success alongside PBR as the primary recovery QA metric."
            ),
        },
        {"id": "localization_chart_block", "type": "chart", "chartId": "localization_bands"},
        {
            "id": "correlation_finding",
            "type": "markdown",
            "sourceId": "diagnostic_aggregate",
            "body": (
                "## Dimmer output pixels do not systematically focus worse\n\n"
                "Across all 16,384 output pixels, measurement mean versus target PBR has Pearson r=0.008 and Spearman rho=-0.0004. Measurement mean versus peak-distance error has Spearman rho=-0.037. "
                "The dimmest and brightest output-pixel quintiles also have similar mean PBR (13.64 versus 13.21) and within-2-px rates (54.6% versus 55.5%).\n\n"
                "**Implication:** spatial illumination nonuniformity is not driving the current failure map. This does not rule out a global SNR penalty, but it makes intensity a weak discriminator of which targets fail."
            ),
        },
        {"id": "quintile_chart_block", "type": "chart", "chartId": "pbr_by_intensity_quintile"},
        {
            "id": "stability_finding",
            "type": "markdown",
            "sourceId": "measurement_current",
            "body": (
                "## Acquisition is complete and temporally stable\n\n"
                "The file has the expected 2,147,483,648 bytes and all 65,536x128x128 samples. There are no blank frames, no exact duplicate groups, and no 12-bit saturation. "
                "Batch means span 54.98-56.55 with only +2.10% first-to-last drift; frame-mean CV is 5.45%. The spatial envelope is also flat (outer-to-center mean ratio 0.999; spatial mean CV 3.12%).\n\n"
                "**Implication:** gross frame loss, replay, saturation, batch discontinuity, and illumination falloff are not leading explanations."
            ),
        },
        {"id": "batch_chart_block", "type": "chart", "chartId": "batch_mean"},
        {
            "id": "scope_definitions",
            "type": "markdown",
            "body": (
                "## Scope, data, and metric definitions\n\n"
                "- **Measurement grain:** one uint16 intensity for each `(probe frame, camera y, camera x)`; shape 65,536x128x128.\n"
                "- **Native DN:** the original Polarized8 sensor code before range scaling into 0-4095 storage.\n"
                "- **Target PBR:** intensity at the requested camera coordinate divided by the report's background estimate.\n"
                "- **Peak distance:** Euclidean distance between the requested coordinate and the brightest captured pixel; this detects misplaced focus that PBR alone can hide.\n"
                "- **Comparison basis:** two preserved active512 summaries plus one legacy-mapping run used only as contextual evidence."
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": "diagnostic_aggregate",
            "body": (
                "## Methodology\n\n"
                "The diagnostic streamed the full 2 GB memmap in 1,000-frame batches, built an exact histogram over stored codes, mapped those codes back to their native Polarized8 values, and computed batch and per-output-pixel means. "
                "It then joined each output pixel to the latest 16,384-point focus CSV and measured Pearson/Spearman associations plus equal-count intensity quintiles. Preserved quality, focus, and reconstruction summaries supplied the cross-run comparison.\n\n"
                "The companion notebook and Python module contain the rerunnable checks; the raw measurement and focus files remain unchanged."
            ),
        },
        {"id": "quality_table_block", "type": "table", "tableId": "quality_checks"},
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## Limitations, uncertainty, and robustness checks\n\n"
                "- No matched dark exposure is available, so the distribution proves low occupancy but cannot directly separate photon shot noise, read noise, and black-level behavior.\n"
                "- The 5-6 August active512 comparison changed acquisition time and potentially uncontrolled optical conditions; it is strong observational evidence, not a randomized exposure A/B.\n"
                "- The legacy run uses a different mapping and is included only to test whether high light is sufficient, not to estimate an exposure effect.\n"
                "- Correlation across output pixels cannot rule out a global SNR ceiling shared by every pixel.\n"
                "- The report diagnoses evidence already saved; it does not verify live camera pixel-format support or physical optical alignment."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## Recommended next steps\n\n"
                "1. **Run a controlled small-pattern exposure sweep.** Use the same 64-512 probes at the current 1.5 ms exposure and at roughly 3x exposure. Because exposure must remain below the DMD picture period, reduce projection rate accordingly (for example, around 200 Hz for 4.5 ms exposure). Compare native median/zero share, reconstruction residual, mean PBR, and within-2-px localization.\n"
                "2. **Capture a matched dark stack.** Same ROI, exposure, gain, trigger mode, and frame count scale; estimate black level and noise before deciding whether `ggs21_dark_level=0` is defensible.\n"
                "3. **Run coordinate impulse tests before another 65,536-frame calibration.** Test corners, center, and a grid to verify x/y order, flips, `target_index=y*width+x`, TM row order, and active512 offsets end to end.\n"
                "4. **Use native higher-bit polarization acquisition if the camera supports it.** Avoid treating 8-to-12-bit numerical scaling as added precision.\n"
                "5. **Adopt joint acceptance gates.** As a pilot target, drive the native median above about 5 DN, zero share below 1%, saturation below 0.1%, and require both target PBR and peak-distance success. Tune thresholds after the exposure sweep and dark measurement."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "- Does the camera expose a native Polarized12/12p format compatible with quadrant extraction?\n"
                "- Do large-miss pixels form a geometric transform pattern (flip, transpose, offset, or tiled displacement) in the saved peak-coordinate map?\n"
                "- How do PBR and localization respond when only exposure changes and the probe set, mapping, optics, and reconstruction seed remain fixed?"
            ),
        },
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "128x128 measurement intensity and recovery diagnosis",
            "description": "Technical diagnosis of whether low measurement intensity explains weak TM recovery.",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": manifest_sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "native_histogram_bins": diagnostic["native_histogram_bins"],
                "batch_profile": diagnostic["batch_profile"],
                "intensity_quintiles": focus["intensity_quintiles"],
                "localization_bands": localization_bands,
                "run_comparison": diagnostic["run_comparison"],
                "quality_checks": quality_checks,
            },
            "accessIssues": [],
        },
        "sources": canonical_sources,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="measurement_intensity_diagnostic_20260806.json"
    )
    parser.add_argument(
        "--output",
        default=(
            "measurement_intensity_report_20260806/"
            "measurement_intensity_report_artifact.json"
        ),
    )
    arguments = parser.parse_args()
    with open(arguments.input, "r", encoding="utf-8") as handle:
        diagnostic = json.load(handle)
    artifact = build_artifact(diagnostic)
    output_directory = os.path.dirname(os.path.abspath(arguments.output))
    os.makedirs(output_directory, exist_ok=True)
    with open(arguments.output, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(os.path.abspath(arguments.output))


if __name__ == "__main__":
    main()
