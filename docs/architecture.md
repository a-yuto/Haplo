# Haplo — アーキテクチャ詳細

## ファイル構成

```
Haplo/
├── Cargo.toml          # 依存: ndarray = "0.16" のみ
├── SPEC.md             # 言語仕様インデックス
├── CLAUDE.md           # このガイドのインデックス
├── .gitattributes      # GitHub(Linguist) で .hpl を Elm としてハイライト
├── docs/               # 詳細ドキュメント
│   ├── spec-goals.md   # §0〜§2, §4, §5
│   ├── spec-syntax.md  # §3.1〜§3.5
│   ├── spec-types.md   # §3.6〜§3.8（北極星サンプル）
│   ├── spec-roadmap.md # §6〜付録
│   ├── architecture.md # このファイル
│   └── p1-plan.md      # P1 実装計画・未完成箇所
├── examples/           # 実行可能な Haplo サンプル（.hpl）
│   ├── functional.hpl      # 関数型スタイル（カリー化・let..in・if・パイプ |>）
│   ├── activations.hpl     # テンソル演算と組み込み（tanh/exp/sqrt/mean・@・ブロードキャスト）
│   ├── linreg_forward.hpl  # 線形回帰の forward + MSE（P0 機能のみ）
│   ├── linreg_train.hpl    # 北極星: 線形回帰の学習ループ（grad + iterate, G3）
│   └── shape_check.hpl     # P2 shape 検査のショーケース（不一致の実行前検出, G4）
└── src/
    ├── ast.rs          # AST 型定義（ロジックなし）
    ├── lexer.rs        # トークナイザ
    ├── parser.rs       # 再帰下降パーサ
    ├── value.rs        # Value / Env / EvalError
    ├── autodiff.rs     # reverse-mode 自動微分テープ（P1）
    ├── interpreter.rs  # ツリーウォーキング評価器
    ├── shape_stage.rs  # shape staging パス（P2 式推論 + P3 型注釈駆動検査）
    └── main.rs         # CLI エントリポイント + run()
```

パイプライン: `lex()` → `parse()` → `shape_eval_program()` → `eval_program()` → `println!`
（`shape_eval_program` は P2 で追加した実行前の shape 検査ゲート。P3 で型注釈駆動の
本体検査（pass3）を追加し、注釈付き関数の引数を宣言 shape に束縛して本体・戻り型を検査する）

---

## 評価器（interpreter.rs）

**ツリーウォーキング**を選んだ理由: P1 の autodiff テープを `eval()` の呼び出し順に記録できるため。バイトコード方式では eval の中間状態にフックを挟みにくい。

- 組み込み関数（`sum`, `mean` 等）は `Value::Builtin` として環境に注入。`eval()` 内で名前を特別扱いしない。これにより |> パイプでも通常どおり使える
- 多引数関数は `desugar_lambda` でカリー化（`f x y = body` → `Lambda{x, Lambda{y, body}}`）。`rev().fold()` の順序に注意（rev なしだと引数順が逆になる）
- 多引数組み込み（`reshape`/`grad`/`iterate`）は `Value::PartialBuiltin` で部分適用を貯め、arity に達したら実行する。`apply` が arity を見て分岐する
- グローバル環境は **two-pass** で構築（P1）。pass1 で関数定義をクロージャ化して共有 globals に登録（前方参照・相互再帰可）、pass2 で値定義をソース順に評価。値定義の前方参照は不可（評価が即時のため）

---

## 自動微分（autodiff.rs — P1）

reverse-mode 自動微分。演算を実行するたびに「演算ノード」を `thread_local` のテープ
（Wengert list）へ順に積み、`backward()` で末尾（出力）から逆向きに随伴を累積する。

- **thread_local テープを選んだ理由**: ツリーウォーキングの `eval()` は AST を深く再帰
  するため、`&mut Tape` を全段に通すとシグネチャ変更が広範囲に及ぶ。thread_local なら
  `eval()` の形を変えずに、grad の評価中だけ記録フックを差し込める
- `grad f x` は、入力 `x` を葉ノードにし、`f` を `Value::Tracked(node_id)` に適用して
  loss を評価する。`eval_binop` と単項組み込み（exp/log/tanh/sqrt/sum/mean）が Tracked を
  検出するとテープへ記録する。出力ノードから backward して入力の随伴を勾配として返す
- スカラーは 0 次元 ArrayD で表す。これでスカラーとテンソルを同じノード型で扱え、
  要素ごと演算の scalar↔tensor ブロードキャストも一様に書ける（`reduce_to` で逆操作）
- `Value::Tracked` は grad のスコープ外には漏れない（通常の評価では生成されない）

---

## 環境（value.rs）

- `Env` は2層構造（P1）。ローカル束縛（引数・let）は Rc 永続連結リスト、トップレベル
  定義は共有 `Rc<RefCell<HashMap>>`（globals）。`lookup` は locals → globals の順に引く
- locals を永続連結リストにする理由: `extend()` が O(1)、複数クロージャが同じ親 env を安全に共有できる
- globals を共有可変マップにする理由: 前方参照・相互再帰のため。全クロージャが同じ
  globals をキャプチャするので、定義後に追加された束縛も呼び出し時に解決できる（knot-tying）
- `Value::Int` と `Value::Float` を分けて保持（整数除算の維持と表示の自然さのため。`6.0` → `"6.0"` と表示）
- テンソルは `Rc<ArrayD<f64>>` でくるむ（クロージャキャプチャ時のコピーを O(1) に抑える）。Arc ではなく Rc を使うのは P0 がシングルスレッドだから

---

## レキサー（lexer.rs）

- `Vec<char>` で保持する理由: `peek2()`（2文字先読み）に `chars[pos+1]` が必要なため
- 改行の意味判定: `open_depth == 0 && !last_was_continuation` のときのみ `Newline` を emit
  - `open_depth`: `(` / `[` の入れ子深度。括弧内は常に行継続
  - `last_was_continuation`: 演算子・コンマ・開き括弧の後は true。行がまだ続くことを示す
- `sum`/`mean` 等の組み込み名はキーワードにしない（`Ident` として通過させ、環境で解決）。将来ユーザが同名の関数を定義できるようにするため

---

## パーサー（parser.rs）

- **手書き再帰下降**（nom/pest/lalrpop 不採用）。エラーメッセージのカスタマイズと、Haplo の文法が小さいことが理由
- 演算子優先順位は**関数呼び出しチェーン**で実装（Pratt パーサ不採用）。`parse_additive` が `parse_matmul` を呼ぶ等、各優先順位レベルが独立した関数として見える
- `next_starts_atom` から `Minus` を除外（`10 - 3` を `App(10, -3)` に誤解釈しないため。負数引数は `f (-3)` と書く）
- `^` は再帰呼び出しで右結合を実現（`2^3^4` → `Pow(2, Pow(3, 4))`）

---

## Shape Staging パス（src/shape_stage.rs — P2 で実装済み）

`interpreter.rs` と対称な設計。`Value` の代わりに `ShapeType` を返す `shape_eval()` を実装した。
評価器と同じ AST を再帰的に歩くため、構造はほぼ鏡像になっている。
カリー化 desugar（`desugar_lambda`）と組み込み arity（`builtin_arity`）は規則のずれを防ぐため
`interpreter.rs` から `pub(crate)` で共有する（shape 評価と実評価で挙動が食い違わないように）。

```
pub fn shape_eval_program(program: &Program) -> Result<ShapeType, ShapeError>

ShapeType:
  Scalar                            // スカラー値
  Tensor(Vec<DimVal>)               // テンソル（次元の列）
  Fn(Box<ShapeType>, Box<ShapeType>) // 関数の shape（引数 → 戻り値）

DimVal:
  Concrete(usize)  // 具体的な次元（例: 3）     ← P2 で対応
  Var(String)      // 次元変数（例: "m"）         ← P4 で単一化を追加（型だけ先に定義）
  Unknown          // 推論不能な次元（エラーにしない）← P4 で活用
```

ShapeType は関数を `Fn(arg, ret)` ではなく `Closure{param, body, env}` で表す（評価器の
Closure と同型）。理由：`App` の規則を「body を param=arg_shape で再評価する」形にするため。
`Fn` 形だと引数 shape を変えて body を再評価できず、カリー化・部分適用を正しく扱えない。

主な shape 規則：

```
Lit(Int|Float|Bool)  → Scalar
TensorLit(rows)      → 全要素 Scalar なら 1行 Tensor([cols]) / 複数行 Tensor([rows,cols])
BinOp(@,  a, b)      → 内次元一致チェック後 Tensor([m, n])（不一致は MatMulMismatch）
BinOp(+, -等, a, b)  → 両辺が完全 Concrete で一致時のみ通過（不一致は ElementwiseMismatch）
App(Closure, arg)    → body を param=arg_shape で再評価
sum/mean             → Scalar、exp/log/tanh/sqrt → 入力と同 shape
zeros/ones/reshape   → Unknown（出力 shape が引数の「値」依存のため断定しない）
grad f x             → x と同 shape、iterate init n f → init と同 shape
```

**偽陽性ゼロ方針**：推論できない箇所はすべて `ShapeType::Unknown` を伝播させ、
「両辺がすべて Concrete で確定し、かつ矛盾している」場合だけをエラーにする。
未定義変数・main 不在・型不一致といった非 shape エラーは `eval` に委ねて shape パスは通す。

**再帰の停止保証**：shape ドメインには実値が無く、`if` の両枝を評価するため、再帰関数の
shape 評価は自然には停止しない（実評価は実値で片枝＋基底ケースに到達するので止まる）。
そこで `thread_local` の2予算で打ち切る：`APPLY_DEPTH`（クロージャ適用のネスト深度、Drop で
復元＝線形再帰のスタック溢れ防止）と `FUEL`（適用の総回数、消費のみ＝分岐再帰の指数爆発防止）。
どちらの上限でも超過時は `Unknown` を返すだけなので、打ち切りが偽陽性を生むことはない。
深度上限はテストスレッド（約2MBスタック）でも安全に辿れる控えめな値にしている
（正常な非再帰ネストは浅いので実害なし）。

パイプライン統合：`main.rs` の `run()` で `eval_program()` の前に
`shape_eval_program(&program)?` を呼ぶ（P2 で追加）。`HaploError::Shape` で報告する。
