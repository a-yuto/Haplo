# Haplo 言語仕様 — ロードマップ・リスク（§6〜付録）

## §6 段階的開発計画

dependent 型は研究レベルなので段階導入する。

| フェーズ | 目標 | 型システム |
|---------|------|-----------|
| P0 | 動く最小処理系（スカラー/テンソルの四則・行列積をインタプリタ実行） | dtype のみ静的、shape は実行時 |
| P1 | リバースモード autodiff のテープ、`grad` で勾配 | 同上 |
| P2 ✅ | shape の抽象評価（staging パス） | 抽象 shape ドメイン |
| P3 ✅ | shape 検査を型に導入（固定次元） | 静的 shape |
| **P4 ✅** | **shape 多態（次元変数の単一化）＋ shape 算術** | **多相 shape** |
| **P5 ✅** | **値依存 shape（DimList・zeros/ones/reshape の具体 shape 推論・置換評価）** | **dependent（部分）** |
| **P6 ✅** | **標準ライブラリ拡充（abs/max_val/min_val/concat/flatten/norm/clip）・エラー改善・concat/flatten shape 推論** | **—** |

P0〜P4 で既に実用的かつ十分意欲的。P5 を最終目標に置きつつ手前で価値を出す。

### P2：shape staging（形状抽象評価）の詳細 ✅ 実装済み

`src/shape_stage.rs` に実装。既存の評価器（`interpreter.rs`）と対称な「shape 評価器」で、
評価ドメインを `Value → ShapeType` に替えて同じ AST の再帰構造を使い回す。
`run()`（`main.rs`）で `eval_program` の前段ゲートとして `shape_eval_program` を呼び、
行列積の内次元不一致・要素ごと演算の shape 不一致を実行前に検出する。
**偽陽性ゼロ方針**：推論できない箇所（`zeros`/`reshape` の出力など）は `Unknown` を伝播させ、
両辺がすべて具体次元（`Concrete`）で確定し矛盾するケースだけをエラーにする。

```
抽象ドメインの型：
  ShapeType = Scalar | Tensor(Vec<DimVal>) | Fn(ShapeType, ShapeType)
  DimVal    = Concrete(usize) | Var(String) | Unknown

主な操作の shape 規則：
  BinOp(+,  Tensor[m,n], Tensor[m,n]) → Tensor[m,n]   （shape 一致が必要）
  BinOp(@,  Tensor[m,k], Tensor[k,n]) → Tensor[m,n]   （内次元 k の一致が必要）
  App(Closure{param,body}, arg_shape)  → body を param=arg_shape で評価
```

P2 では固定次元（`Concrete` のみ）を対象とし、次元変数の単一化は P4 で導入する。
staging pass の構造を P2 で確立することで、P4 では単一化アルゴリズムを追加するだけでよい。

### P6：標準ライブラリ拡充・エラー改善・concat/flatten shape 推論の詳細 ✅ 実装済み

**新組み込み関数（`value.rs` / `interpreter.rs` / `shape_stage.rs`）**

| 関数 | シグネチャ（概念） | shape 規則 |
|------|-------------------|------------|
| `abs` | `a -> a` | 入力と同 shape |
| `max_val` | `Tensor[...] -> f32` | スカラー（集約） |
| `min_val` | `Tensor[...] -> f32` | スカラー（集約） |
| `concat` | `Tensor[m] -> Tensor[n] -> Tensor[m+n]` | Concrete 時 m+n を確定 |
| `flatten` | `Tensor[m,n] -> Tensor[m*n]` | Concrete 時 m*n を確定 |
| `norm` | `Tensor[...] -> f32` | スカラー（L2 ノルム） |
| `clip` | `f32 -> f32 -> Tensor[...] -> Tensor[...]` | 入力と同 shape |

**shape 算術の実動作（P6 の中核）**

`concat` / `flatten` の shape 規則を `apply_shape_builtin` に実装したことで、
P4 で AST に持たせるだけだった shape 算術（`m+n`, `m*n`）が初めて実際の推論で使われるようになった:

- `concat [1,2] [3,4,5]` → `Tensor[2] concat Tensor[3]` → `Tensor[5]`（2+3=5 を確定）
- `flatten [[1,2,3],[4,5,6]]` → `Tensor[2,3]` → `Tensor[6]`（2*3=6 を確定）

Concrete でない場合（Var/Unknown が絡む）は Unknown にフォールバックして偽陽性ゼロを維持する。

**エラーメッセージ改善**

- `HaploError::Shape` の `Display` に「ヒント」行を追加（型注釈の見直しを促す）
- CLI エラー出力を `error[ファイル名]: ...` 形式に統一（IDE からのジャンプを想定）
- 未使用の `all_concrete` ヘルパを削除（P4 の `dim_pair_conflict` 移行で不要になった）

**サンプル追加: `examples/stdlib_showcase.hpl`**

新組み込み関数 7 本の組み合わせ例。`abs → clip → norm` のパイプラインで信号を正規化する。

### P5：値依存 shape（DimList・置換評価）の詳細 ✅ 実装済み

**DimList による具体 shape 伝播**

P4 まで `zeros`/`ones`/`reshape` は常に `Unknown` を返していた。P5 では整数リテラルのみからなる
1D テンソル（`[3]`、`[2, 4]` など）を `ShapeType::DimList(Vec<usize>)` として認識し、
`apply_shape_builtin` で消費して具体的な `Tensor[Concrete(...)]` を返す。

- `shape_eval_tensor_lit`: 1行かつ全要素が非負整数リテラル → `DimList` を返す
  （浮動小数 `[3.0]` は対象外。偽陽性ゼロ）
- `apply_shape_builtin`:
  - `Zeros | Ones`: `DimList(dims)` → `Tensor[Concrete(n) for n in dims]`
  - `Reshape`: 第2引数が `DimList(dims)` → `Tensor[Concrete(n) for n in dims]`

これにより `zeros [3]` が `Tensor[3]` に、`zeros [2, 4]` が `Tensor[2, 4]` に確定し、
下流の演算でも shape が伝播する。例: `zeros [3] + [1,2]` が `ElementwiseMismatch` を検出できる。

**shape 算術の置換評価（`Subst`）**

`check_annotated_fn` で引数の宣言 `TypeDim::Var(name)` と推論 `DimVal::Concrete(n)` を
突き合わせて置換表 `Subst = HashMap<String, usize>` を構築する。
戻り型を `shape_of_type_with_subst` で変換する際、`TypeDim::Expr` に含まれる変数を
`eval_dim_expr` で評価して `DimVal::Concrete` に解決する。

```
例: double_size : Tensor[n] -> Tensor[n*2]
  引数が Tensor[Concrete(3)] → subst = {n: 3}
  戻り型 Tensor[n*2] → eval_dim_expr(n*2, {n:3}) = 6 → Tensor[Concrete(6)]
```

`subst` に変数が無い場合（引数が Unknown/Var）は `None` を返し `DimVal::Unknown` に
フォールバックするため、偽陽性は生じない。

```
新規追加:
  ShapeType::DimList(Vec<usize>)   -- 整数リテラルの 1D テンソル（shape 指定子）
  type Subst = HashMap<String, usize>
  fn eval_dim_expr(e: &DimExpr, subst: &Subst) -> Option<usize>
  fn shape_of_type_with_subst(ty: &TypeExpr, subst: &Subst) -> ShapeType
  fn build_subst_from_params(param_shapes: &[ShapeType], declared_arrow: &TypeExpr) -> Subst
```

### P4：次元変数の単一化・shape 算術の詳細 ✅ 実装済み

P3 では次元変数（`Var`）を保持・伝播するだけで単一化はしなかった。P4 では以下を実装した：

**次元変数の単一化（`VarConflict` 検出）**

- 同名変数（例: `Tensor[n] + Tensor[n]`）→ 一致が保証される。P4 では `elementwise_shape` が
  `dim_pair_conflict` を使い、同名変数を正しく「等しい次元」として扱い `Tensor[n]` を返す
  （P3 では all_concrete が None のため Unknown を返していた）。
- 異名変数（例: `Tensor[n] + Tensor[m]`）→ 独立した型変数なので等しい保証がない。
  `VarConflict` エラーを報告する。同様に行列積の内次元が異名変数の場合も `VarConflict`。
- ランク不一致（例: `Tensor[n] + Tensor[n, m]`）→ 変数を含む場合でも常に `ElementwiseMismatch`。

`check_annotation` も P4 で拡張した：ランク不一致・宣言 Var 名と推論 Var 名の不一致も
`AnnotationMismatch` として報告する（P3 は完全 Concrete 同士のみ）。

**`->` の結合規則修正**

型式パーサの `parse_type_expr` が左結合で `->` をパースしていたため、
`A -> B -> C` が `Arrow(Arrow(A,B), C)` になっており、多引数注釈で先頭の引数が
`Arrow(A,B)` 全体になって `shape_of_type` が `Unknown` を返す潜在バグがあった。
P4 で右結合（再帰 `parse_type_expr`）に修正し、`A -> (B -> C)` として正しく
`decompose_arrow` で個別引数 shape に剥がせるようにした。

**shape 算術（`TypeDim::Expr`）**

型注釈中に `m+n`, `m*n`, `m+n-1` 等の算術式を書けるようにした。
- AST: `TypeDim::Expr(DimExpr)` 追加。`DimExpr = Lit | Var | Add | Sub | Mul`。
- パーサ: `parse_dim_expr` / `parse_dim_term` / `parse_dim_atom` で Tensor 次元をパース。
  優先順位: `*` > `+/-`（標準算術と同じ）。
- `shape_of_type` では算術式を `DimVal::Unknown` に変換（偽陽性ゼロ）。
  `concat/flatten` 等プリミティブの追加後（P6 目標）に実際の評価を実装予定。

```
エラー種別（P4 で追加/拡張）:
  VarConflict { op, var_a, var_b }  -- 異名変数が要素ごと演算/行列積内次元で衝突
  ElementwiseMismatch               -- ランク不一致でも報告（Var を含む場合も）
  AnnotationMismatch                -- ランク不一致・Var 名不一致も対象に拡張
```

### P3：型注釈駆動の shape 検査（固定次元）の詳細 ✅ 実装済み

P2 の staging パスは式（リテラル）からのボトムアップ推論のみで、関数引数はリテラルが
無いと `Unknown` になり本体が検査されない穴があった。P3 はこの穴を型注釈で塞ぐ。
`src/shape_stage.rs` の `build_shape_env` に **pass3** を追加し、型注釈付き関数の引数を
**宣言 shape に束縛**してから本体を `shape_eval` する。これにより `main` から呼ばれない
関数でも本体の固定次元矛盾を検出できる。あわせて以下を実装した：

- パーサが次元変数名を保持（`TypeExpr::Tensor(Vec<TypeDim>)`、`TypeDim = Fixed | Var`）。
  P2 までは `Vec<Option<usize>>` で変数名を捨てていた。
- `shape_of_type`：型式 → 抽象 shape（`f32` 等→`Scalar`、`Tensor[..]`→`Tensor`で
  `Fixed→Concrete`/`Var→DimVal::Var`、関数型→`Unknown`）。
- `decompose_arrow`：関数型注釈を「引数 shape 列」と「戻り shape」に分解（arity 個 Arrow を剥がす）。
- 注釈付きグローバル値は宣言 shape で登録し、推論不能（`Unknown`）な値でも下流の検査が効く。
- 宣言 shape と本体推論が**両方 `Concrete` で食い違う**ときだけ `AnnotationMismatch`（偽陽性ゼロ）。

P3 は固定次元（`Concrete`）の検査に絞る。次元変数（`Var`）は注釈から保持・伝播するが、
**単一化はしない**（`[m,k] @ [k,n]` の `k` 一致検証や `m+n` 等の shape 算術は P4）。

---

## §7 達成マイルストーン（「動くもの」の目標）

北極星は **G3「勾配降下で loss が下がる」**（§3.8 のサンプルが走ること）。

| 目標 | 達成条件 | 証明されること | フェーズ |
|------|----------|----------------|---------|
| **G0 スカラー電卓** | `main = 2 + 3 * 4` → `14` | lexer→parser→評価のパイプライン疎通 | P0 ✅ |
| **G1 テンソル電卓** | `[1,2,3] + [4,5,6]`、`a @ b`、`sum v` が動く | ndarray 連携・演算子 | P0 ✅ |
| **G2 微分が動く** | `grad (\w -> sum (w^2))` に `[1,2,3]` → `[2,4,6]` | autodiff テープ | P1 ✅ |
| **G3 学習が回る** | 線形回帰サンプルが走り、`iterate` で loss が下がる | スタック全体 | P1 ✅ |
| **G4 staging** | `shape_eval` パスが通り `[2,3] @ [2,2]` の不一致を実行前に報告 | 抽象評価パイプラインの疎通 | P2 ✅ |
| **G5 型が守る** | `(3,4) @ (3,4)` の不一致がコンパイル時エラー＋分かりやすいメッセージ | 静的 shape 検査（依存型の価値） | P3（固定次元）✅〜P5 |

**戦略のコツ**：まず動的型（shape は実行時チェック）で G3 まで到達。次に G4（staging パス）で shape 検査の基盤を作り、最後に静的型システム（G5）を足す。

**G3 までに必要な最小機能**：f32 のスカラーとテンソル、`+ - * / ^ @`、`sum`/`mean`、`grad`、`iterate`（なければ再帰）、`let`、関数定義。

**G3 までは後回しでよい機能**：依存型・`case`・型別名・`fold`・複雑なブロードキャスト・スライス。

---

## §8 技術的リスク・難所

1. 依存型の型検査：次元の単一化、shape 算術（`m+n`）の等価判定、値依存型の健全性。最大の難所
2. shape 推論の実装コストと推論範囲のトレードオフ
3. autodiff と型の整合（勾配の shape が元関数から正しく導かれること）
4. 純粋言語での IO 設計（出力をどう扱うか）
5. スコープ管理：各フェーズで必ず「動くもの」を出す規律
6. shape staging パスの健全性：`shape_eval` が実際の `eval` と一致するか。テンソルリテラル・カリー化・クロージャキャプチャで食い違うと「staging は通るが実行時に shape エラー」が起きうる。同一テストケースで両者を並走させて確認する

---

## §9 未決定・要確認の論点

- GPU 対応の要否（当面は CPU/ndarray 想定でよいか）
- IO（出力）の設計方針（純粋性をどう保つか）
- ブロードキャスト規則の詳細仕様
- スライス構文（`A[0:2, :]` 等）の導入時期
- dtype の明示記法（既定 f32 ＋ `f64`/`Int` でよいか）
- 高階微分・forward-mode の将来対応有無
- 標準ライブラリの初期スコープ（NN 層・最適化器まで入れるか）
- REPL / `import`（モジュール分割）の導入時期

---

## 付録：参考になる既存言語・実装

- **Dex**（Google）：依存型で shape を扱う ML 向け研究言語。最も近い先行例
- **Futhark**：配列特化・型でサイズを扱う関数型言語
- **JAX / Autograd**：reverse-mode autodiff の設計参考
- **Elm**：文法スタイルの参考
- **dfdx / candle / burn**（Rust）：Rust でのテンソル＋autodiff 実装の構造参考
