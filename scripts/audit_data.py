"""Data audit across the delivered CSVs - the first thing a careful researcher does
before any modelling. Prints a compact report, writes data/11_audit_report.csv.
"""
import os
import sys

import pandas as pd

D = r"C:\Users\Administrator\Desktop\大创申报\data"


def load(name):
    return pd.read_csv(os.path.join(D, name))


def line(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    mp = load("09_mp_magnetism.csv")
    opt = load("01_dft_structures_mp.csv")
    lit = load("03_literature_records.csv")
    tab = load("05_literature_tables.csv")
    rec = []

    line("1. 两个 DFT 池子的规模与目标可用性")
    for pool in ("design_space", "context"):
        sub = mp[mp["pool"] == pool]
        print(f"{pool:<14} n={len(sub):<6} 有 Ms_T={sub['Ms_T'].notna().sum():<6} "
              f"FiM/FM={sub['ordering'].isin(['FM', 'FiM']).sum():<6} "
              f"AFM/NM={sub['ordering'].isin(['AFM', 'NM']).sum()}")
        rec.append({"check": f"{pool} 行数", "value": len(sub)})
        rec.append({"check": f"{pool} 磁有序构成",
                    "value": dict(sub["ordering"].value_counts())})

    line("2. 多形体（同一化学式多条记录）——标签噪声的主要来源")
    dup = mp[mp.duplicated("formula_pretty", keep=False)]
    nun = dup["formula_pretty"].nunique()
    print(f"重复化学式条目数={len(dup)}  涉及化学式={nun} 个 "
          f"（占全部 {len(mp)} 条的 {len(dup)/len(mp):.1%}）")
    ex = (mp[mp["formula_pretty"] == "Nd2Fe14B"]
          [["formula_pretty", "material_id", "Ms_T", "energy_above_hull_eV", "is_stable"]])
    print("\n以 Nd2Fe14B 为例：")
    print(ex.to_string(index=False))
    rec.append({"check": "重复化学式条目", "value": len(dup)})

    line("3. 两个 MP 通道的重叠（OPTIMADE 无密钥 vs API 带密钥）")
    o_ids = set(opt["mp_id"])
    m_ids = set(mp["material_id"])
    print(f"OPTIMADE 条目={len(o_ids)}  API 条目={len(m_ids)}  交集={len(o_ids & m_ids)}")
    print("→ 交集部分若同时使用，等于把同一份 DFT 结果算了两遍")
    rec.append({"check": "OPTIMADE∩API", "value": len(o_ids & m_ids)})

    line("4. 设计空间 vs 背景池的成分重叠（迁移学习的泄漏风险）")
    ds_sys = set(mp[mp["pool"] == "design_space"]["chemsys"])
    ctx_sys = set(mp[mp["pool"] == "context"]["chemsys"])
    print(f"设计空间化学体系={len(ds_sys)}  背景池={len(ctx_sys)}  "
          f"重合体系={len(ds_sys & ctx_sys)}")
    print(f"重合比例 {len(ds_sys & ctx_sys)/max(len(ds_sys),1):.1%} "
          "→ 微调前必须按体系做分组切分，不能随机切分")
    rec.append({"check": "化学体系重合", "value": len(ds_sys & ctx_sys)})

    line("5. 目标量分布（是不是长尾、有没有离群）")
    for pool in ("design_space", "context"):
        s = mp.loc[mp["pool"] == pool, "Ms_T"].dropna()
        if len(s):
            q = s.quantile([0, .25, .5, .75, 1]).round(3).tolist()
            print(f"{pool:<14} Ms_T min/q25/中位/q75/max = {q}")
    nm = mp[(mp["ordering"].isin(["NM", "AFM"])) & (mp["Ms_T"].fillna(0) > 0.3)]
    print(f"\n有序类型为 NM/AFM 却算得 Ms_T>0.3 T 的矛盾条目：{len(nm)} 条")
    rec.append({"check": "NM/AFM 但 Ms 偏高", "value": len(nm)})

    line("6. 缺失情况（背景池哪些列不能直接用）")
    miss = mp[["Ms_T", "energy_above_hull_eV", "density_g_cm3",
               "space_group_number", "formation_energy_per_atom_eV"]].isna().mean()
    print((miss * 100).round(2).to_string())

    line("7. 文献抽取层的单位分布与重复")
    print(lit.groupby(["property", "unit_SI"]).size().to_string())
    print(f"\n同一组分+同一性能出现多次的记录对："
          f"{lit.duplicated(['composition_nearest', 'property']).sum()} 条")
    print(f"置信度分布：{dict(lit['confidence'].value_counts())}")
    rec.append({"check": "文献记录数", "value": len(lit)})

    line("8. 表格层的可用密度")
    mag = tab[tab["is_magnetic_table"] == True]  # noqa: E712
    print(f"表格总行={len(tab)}  判定含磁性的行={len(mag)}  "
          f"其中提到 Nd-Fe-B={int(tab['mentions_NdFeB'].sum())}")

    pd.DataFrame(rec).to_csv(os.path.join(D, "11_audit_report.csv"),
                             index=False, encoding="utf-8-sig")
    print("\nWROTE 11_audit_report.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
