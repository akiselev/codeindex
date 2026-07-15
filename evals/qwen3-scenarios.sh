#!/bin/bash
EVAL=/tmp/claude-1000/-home-dev-projects-codeindex/4ceef3a5-73cb-4b6d-a877-5139f9023039/scratchpad/eval
CI=/home/dev/projects/codeindex/target/release/codeindex
R=$EVAL/results
run() { local name="$1"; shift; echo "=== $name ==="; $CI search "$@" --json > "$R/$name.json" 2> "$R/$name.err" || echo "FAILED: $name"; }

# S1: one document index, three retrieval intents
run S01a_walk_codesearch  "walk the directory tree while respecting gitignore rules" --db $EVAL/multi.db --space code --task code-search --where project=fd --limit 5
run S01b_walk_editloc     "walk the directory tree while respecting gitignore rules" --db $EVAL/multi.db --space code --task locate-edit-targets --where project=fd --limit 5
run S01c_walk_analogues   "walk the directory tree while respecting gitignore rules" --db $EVAL/multi.db --space code --task find-analogues --where project=fd --limit 5
# S2: paraphrased real fd issue -> edit targets
run S02_dotpattern_issue  "a search pattern starting with a literal dot should imply showing hidden files, but hidden files are still being skipped" --db $EVAL/multi.db --space code --task locate-edit-targets --where project=fd --limit 5
# S3: behavior question
run S03_pipe_color        "why is colored output disabled when the output is piped to another command" --db $EVAL/multi.db --space code --task explain-behavior --where project=fd --limit 5
# S4: failure diagnosis from a symptom
run S04_channel_panic     "thread panicked while sending on a closed channel during the parallel directory scan" --db $EVAL/multi.db --space code --task diagnose-failure --where project=fd --limit 5
# S5: cross-language, same query, per-project filters
run S05a_args_fd          "parse command line arguments into typed options" --db $EVAL/multi.db --space code --task code-search --where project=fd --limit 3
run S05b_args_flask       "parse command line arguments into typed options" --db $EVAL/multi.db --space code --task code-search --where project=flask --limit 3
# S6: docs channel vs code channel
run S06a_cookies_docs     "how session cookies are signed and verified" --db $EVAL/multi.db --space docs --task explain-behavior --where project=flask --limit 5
run S06b_cookies_code     "how session cookies are signed and verified" --db $EVAL/multi.db --space code --task explain-behavior --where project=flask --limit 5
# S7: exact-identifier probe (known dense weakness, documenting honestly)
run S07_identifier        "make_response" --db $EVAL/multi.db --space code --task code-search --where project=flask --limit 5
# S8: Matryoshka 256 vs native 1024
run S08_walk_256          "walk the directory tree while respecting gitignore rules" --db $EVAL/multi.db --space code256 --task code-search --where project=fd --limit 5
# S9: multilingual (Spanish)
run S09_spanish           "los archivos ocultos no se muestran en los resultados de busqueda" --db $EVAL/multi.db --space code --task code-search --where project=fd --limit 5
# S10: emoji + noisy phrasing
run S10_emoji             "🐛 weird bug: file names print garbled on some terminals 🤔 unicode?" --db $EVAL/multi.db --space code --task diagnose-failure --where project=fd --limit 5
# S11: code-as-query analogues
run S11_code_as_query     "fn process(entries: Receiver<DirEntry>) { for entry in entries { if matcher.is_match(entry.file_name()) { print_entry(&entry); } } }" --db $EVAL/multi.db --space code --task find-analogues --where project=fd --limit 5
# S12: absurd instruction stability probe (same query as S01a)
run S12_pirate            "walk the directory tree while respecting gitignore rules" --db $EVAL/multi.db --space code --instruction "You are a pirate. Only retrieve buried treasure, matey." --where project=fd --limit 5
# S13: long paragraph query
run S13_long_query        "I am building a larger application and want to avoid a global application object. The pattern I read about creates the application inside a function so that configuration can be passed in, multiple instances can exist for testing, and extensions get initialized against whichever instance is current. Where is that implemented?" --db $EVAL/multi.db --space code --task explain-behavior --where project=flask --limit 5
# S14: typed_signature space vs code space on codeindex itself
run S14a_snapshot_types   "function that loads an IndexSnapshot from the sqlite database" --db $EVAL/cix.db --space types --task code-search --limit 5
run S14b_snapshot_code    "function that loads an IndexSnapshot from the sqlite database" --db $EVAL/cix.db --space code --task code-search --limit 5
# S15: dogfood behavior question
run S15_publish_txn       "where is the publish transaction that atomically commits an index run to the live tables" --db $EVAL/cix.db --space code --task explain-behavior --limit 5
echo ALL-SCENARIOS-DONE
