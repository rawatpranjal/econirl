"""Generate tex table fragments from primer result JSONs.

Reads JSON files from papers/primers/tables/*.json and writes
corresponding .tex table fragments that primers can \\input.

Usage:
    python papers/primers/generate_tables.py
"""

import json
from pathlib import Path


TABLES_DIR = Path(__file__).parent / "tables"


def generate_nfxp_table(data: dict, outpath: Path):
    """Write NFXP results as a tex tabular fragment."""
    true = data["true_params"]
    est = data["estimated_params"]
    se = data["standard_errors"]

    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Parameter & True & Estimate & SE \\",
        r"\midrule",
        f"Operating cost ($\\theta_c$) & {true['operating_cost']} "
        f"& {est['operating_cost']} & ({se['operating_cost']}) \\\\",
        f"Replacement cost ($RC$) & {true['replacement_cost']:.1f} "
        f"& {est['replacement_cost']} & ({se['replacement_cost']}) \\\\",
        r"\midrule",
        f"Log-likelihood & & \\multicolumn{{2}}{{r}}{{{data['log_likelihood']}}} \\\\",
        f"Policy accuracy & & \\multicolumn{{2}}{{r}}"
        f"{{{100 * data['policy_accuracy']:.1f}\\%}} \\\\",
        f"Time (seconds) & & \\multicolumn{{2}}{{r}}{{{data['time_seconds']}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]

    outpath.write_text("\n".join(lines) + "\n")
    print(f"  Wrote {outpath}")


def generate_tdccp_table(data: dict, outpath: Path):
    """Write TD-CCP results as a tex tabular comparing to NFXP."""
    true = data["true_params"]
    r = data["results"]
    nfxp = r["nfxp"]
    tdccp = r["tdccp"]

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"& True & NFXP & TD-CCP & TD-CCP SE \\",
        r"\midrule",
        f"Operating cost ($\\theta_c$) & {true['operating_cost']} "
        f"& {nfxp['operating_cost']} & {tdccp['operating_cost']} "
        f"& ({tdccp['se_oc']}) \\\\",
        f"Replacement cost ($RC$) & {true['replacement_cost']:.1f} "
        f"& {nfxp['replacement_cost']} & {tdccp['replacement_cost']} "
        f"& ({tdccp['se_rc']}) \\\\",
        r"\midrule",
        f"Log-likelihood & & {nfxp['ll']} "
        f"& \\multicolumn{{2}}{{r}}{{{tdccp['ll']}}} \\\\",
        f"Time (seconds) & & {nfxp['time']} "
        f"& \\multicolumn{{2}}{{r}}{{{tdccp['time']}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]

    outpath.write_text("\n".join(lines) + "\n")
    print(f"  Wrote {outpath}")


def generate_generic_table(data: dict, outpath: Path):
    """Generic fallback table generator for any estimator."""
    est = data.get("estimated_params", {})
    true = data.get("true_params", {})
    se = data.get("standard_errors", {})

    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Parameter & True & Estimate & SE \\", r"\midrule"]
    for key in est:
        t = true.get(key, "")
        e = est[key]
        s = se.get(key, "")
        s_str = f"({s})" if s != "" else ""
        lines.append(f"{key} & {t} & {e} & {s_str} \\\\")
    lines += [r"\midrule",
              f"Log-likelihood & & \\multicolumn{{2}}{{r}}"
              f"{{{data.get('log_likelihood', '')}}} \\\\",
              f"Time (seconds) & & \\multicolumn{{2}}{{r}}"
              f"{{{data.get('time_seconds', '')}}} \\\\",
              r"\bottomrule", r"\end{tabular}"]

    outpath.write_text("\n".join(lines) + "\n")
    print(f"  Wrote {outpath}")


def main():
    json_files = sorted(TABLES_DIR.glob("*_results.json"))
    if not json_files:
        print("No result JSONs found in", TABLES_DIR)
        return

    for jf in json_files:
        data = json.loads(jf.read_text())
        estimator = data.get("estimator", "").lower().replace("-", "_").replace(" ", "_")
        tex_path = TABLES_DIR / f"{jf.stem.replace('_results', '')}_results.tex"

        if "nfxp" in estimator or "nfxp" in jf.stem:
            generate_nfxp_table(data, tex_path)
        elif "td_ccp" in estimator or "tdccp" in jf.stem:
            generate_tdccp_table(data, tex_path)
        else:
            generate_generic_table(data, tex_path)

    print("Done.")


if __name__ == "__main__":
    main()
