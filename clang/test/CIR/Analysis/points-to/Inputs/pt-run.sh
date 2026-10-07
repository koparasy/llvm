#!/usr/bin/env bash
# Run a C++ program with its points-to NoAlias facts checked at run time.
#
#   pt-run.sh <file.cpp> <out-prefix> [instrument options...]
#
# Steps: clang++ -fclangir -emit-cir -O0, cir-opt -test-cir-points-to-instrument
# (options such as closed-tu=1 max-pairs-per-func=0 seed=2), cir-translate
# --cir-to-llvmir, compile and link with the runtime, run. The side table is
# <out-prefix>.table. On a violation the table line of the pair is printed.
# Tools come from $BIN (default: the directory of `which cir-opt`).
# $PT_EXTRA lists more sources, compiled without CIR or checks and linked in:
# code the analysis does not see. $CXXFLAGS goes to every C++ compile.
# If $PT_SELF_EXTERNAL is set (for example to -DPT_EXTERNAL), <file.cpp> is
# also compiled that way, without CIR or checks, and linked in: one source
# can hold both the analysed program and its external part.
set -u
src=$1; out=$2; shift 2
case $out in /*) ;; *) out=$PWD/$out ;; esac
here=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-$(dirname "$(command -v cir-opt)")}
CXXFLAGS=${CXXFLAGS:-}
PT_EXTRA=${PT_EXTRA:-}
PT_SELF_EXTERNAL=${PT_SELF_EXTERNAL:-}
opts="table=$out.table $*"
"$BIN/clang++" -fclangir -emit-cir -O0 $CXXFLAGS "$src" -o "$out.cir" || exit 2
"$BIN/cir-opt" "$out.cir" "-test-cir-points-to-instrument=$opts" \
  -o "$out.inst.cir" || exit 3
"$BIN/cir-translate" --cir-to-llvmir "$out.inst.cir" -o "$out.ll" || exit 4
# The CIR lowering spells the data layout differently from the target's
# default (same layout, more explicit entries); the backend rejects that, so
# let clang use the target's.
sed -i '/^target datalayout/d' "$out.ll"
"$BIN/clang" -O0 -c "$here/pt_check_runtime.c" -o "$out.rt.o" || exit 5
extra=()
for f in $PT_EXTRA; do
  obj="$out.extra.$(basename "$f").o"
  "$BIN/clang++" -O0 $CXXFLAGS -c "$f" -o "$obj" || exit 5
  extra+=("$obj")
done
if [ -n "$PT_SELF_EXTERNAL" ]; then
  "$BIN/clang++" -O0 $CXXFLAGS $PT_SELF_EXTERNAL -c "$src" \
    -o "$out.self-external.o" || exit 5
  extra+=("$out.self-external.o")
fi
"$BIN/clang++" -O0 "$out.ll" "$out.rt.o" "${extra[@]}" -o "$out.exe" || exit 6
"$out.exe" 2> "$out.stderr" > "$out.stdout"
status=$?
cat "$out.stderr" >&2
id=$(sed -n 's/^PT-VIOLATION \([0-9]*\)$/\1/p' "$out.stderr")
if [ -n "$id" ]; then
  echo "violated pair:" >&2
  grep "^$id " "$out.table" >&2
fi
exit $status
