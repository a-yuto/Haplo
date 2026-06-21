# Haplo — P1 実装計画・未完成箇所・規約

## 開発フェーズ

| フェーズ | 状態 | 主な作業 |
|---------|------|---------|
| P0 | **完了** | lexer / parser / インタプリタ（スカラー+テンソル） |
| P1 | **完了** | autodiff テープ、`grad`、`iterate`、前方参照 |
| P2 | **次** | 静的 shape 検査（固定次元） |
| P3 | 未着手 | 次元変数の単一化・shape 算術 |
| P4 | 未着手 | 完全な dependent 型 |

---

## P1 で実装した最小セット（G3 達成） ✅

G3 = 線形回帰サンプル（`docs/spec-types.md` §3.8）が走ること。**達成済み**。
北極星プログラムは `examples/linreg_train.hpl` で実行できる。

1. **`Tape`（autodiff テープ）** → `src/autodiff.rs` に実装 ✅
   - reverse-mode 自動微分。`eval()` の呼び出し順に演算ノードを `thread_local` の
     テープへ記録し、`backward()` で逆向きに随伴を累積して勾配を出す。
   - スカラーは 0 次元 ArrayD で表現。要素ごと演算は scalar↔tensor ブロードキャスト対応。
2. **`grad` 組み込み**：`grad : (Tensor -> f32) -> Tensor -> Tensor` ✅
   - `Value::Tracked(usize)` でテープノードを指し、`eval_binop`／単項組み込みが
     Tracked を検出するとテープへ記録する。
3. **`iterate` 組み込み**：`iterate : a -> Int -> (a -> a) -> a` ✅（ループで実装）
4. **前方参照・相互再帰**（two-pass スコープ構築） ✅
   - `Env` に共有 globals マップ（`Rc<RefCell<HashMap>>`）を追加。
   - pass1：関数定義をクロージャ化して globals に登録（相互再帰・前方参照可）。
   - pass2：値定義をソース順に評価。

実装メモ：`eval()` のシグネチャは変えず、テープを `thread_local` に置くことで
記録フックを差し込んだ（`&mut Tape` を全段に通す案より変更が小さい）。

多引数組み込み（`reshape`/`grad`/`iterate`）は `Value::PartialBuiltin` で部分適用を
貯め、arity に達したら実行する仕組みにした。

---

## 未完成箇所の状況

| 箇所 | 状態 | 対応予定 |
|------|------|---------|
| `reshape` | **実装済み**（要素数チェック後に変形） | — |
| `iterate` | **実装済み**（ループ） | — |
| 前方参照・相互再帰 | **実装済み**（two-pass + 共有 globals） | — |
| スカラー左 `Div`/`Pow` | **実装済み**（`1.0 / t`, `s ^ t`） | — |
| `case` 式 | AST に未定義 | P2 |
| `fold` | 未実装 | P2 |
| 値定義の前方参照 | 関数のみ対応（値はソース順評価） | 仕様として許容 |
| パイプ `|>` の複数行 | 同一行のみ | P2 |
| shape 静的検査 | 実行時チェックのみ | P2 |

---

## サンプル検証で判明した SPEC との差分（`examples/` 実行時）

`examples/*.hpl` を P0 インタプリタで走らせて確認した、SPEC（§3.4）に未達の挙動。
いずれも回避策はあるが、SPEC どおりに書けるようにするための課題。

| 箇所 | 現状 | SPEC との差分 | 対応予定 |
|------|------|--------------|---------|
| スカラー ÷ / ^ テンソル | **P1 で対応済み**（`1.0 / t`, `s ^ t`） | — | 完了 |
| `1D @ 2D` 行列積 | `@` は 2D×2D / 2D×1D のみ。1D×2D は明示エラー | 行ベクトル（`Tensor[3]`）を重み行列に掛けられない。2D で書く必要あり | P2 |
| パイプ `|>` の改行 | 同一行のみ解釈。改行をまたぐと `Pipe` が予期しないトークン扱い | §3.4 のパイプを複数行に分けて書けない（パーサが改行を区切りと誤認） | P2 |

注: `Add`/`Sub`/`Mul` はスカラー左右どちらのブロードキャストも対応済み。`Div`/`Pow` も P1 で対応した。

---

## コーディング規約

- コメントは `//`（非公開コード）または `///`（公開 API）
- テストブロック `#[cfg(test)]` へのコメントは不要
- 外部クレートは最小限（現在 `ndarray` のみ）
- `unwrap()` はテスト内と「到達不能」が証明できる箇所のみ許容
- エラーは `HaploError` に統合（`anyhow`/`thiserror` 不採用、外部依存最小化）

---

## テスト構成（現在 75 本）

```
cargo test                       # 全テスト
cargo test g0                    # G0 スカラーテスト（interpreter.rs）
cargo test g1                    # G1 テンソルテスト（interpreter.rs）
cargo test p1                    # P1 grad/iterate/前方参照/reshape（interpreter.rs）
cargo test g3                    # G3 北極星プログラム（interpreter.rs）
cargo test integration           # 統合テスト（main.rs）
```

テストは各ソースファイル末尾の `#[cfg(test)]` ブロックにある。
インタプリタのテストが最多（`g0_*` スカラー 15 本、`g1_*` テンソル 10 本、
`p1_*` autodiff/iterate/前方参照 ほか、`g3_*` 線形回帰の学習ループ）。
