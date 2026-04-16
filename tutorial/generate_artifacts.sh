#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
lesson_name="${1:-Lesson-01}"
lesson_dir="$script_dir/$lesson_name"
clang="$repo_root/build-orig/bin/clang++"
build_include="$repo_root/build-orig/include"

if [[ ! -d "$lesson_dir" ]]; then
  echo "Unknown lesson directory: $lesson_dir" >&2
  exit 1
fi

source_file="$(find "$lesson_dir" -maxdepth 1 -name '*.cpp' | sort | head -n 1)"
if [[ -z "$source_file" ]]; then
  echo "No tutorial source file found in $lesson_dir" >&2
  exit 1
fi

base_name="$(basename "$source_file" .cpp)"
filter_name="${2:-$(sed -n 's/^void \(submit_[A-Za-z0-9_]*\).*/\1/p' "$source_file" | head -n 1)}"
if [[ -z "$filter_name" ]]; then
  echo "Could not infer submit_* function name from $source_file" >&2
  exit 1
fi

out_dir="$lesson_dir/out"
header_file="$out_dir/$base_name.hpp"
footer_file="$out_dir/${base_name}_footer.hpp"
host_wrapper="$out_dir/$base_name.host_wrapper.cpp"

mkdir -p "$out_dir"

cat > "$host_wrapper" <<EOF
#include "$source_file"
#include "$header_file"
#include "$footer_file"
EOF

conda run -n llvm-dev "$clang" -std=c++17 -fsycl-device-only -fsyntax-only \
  -I "$lesson_dir" \
  -Xclang -ast-dump \
  -Xclang -fsycl-int-header="$header_file" \
  -Xclang -fsycl-int-footer="$footer_file" \
  "$source_file" > "$out_dir/$base_name.device.ast.txt" \
  2> "$out_dir/$base_name.device.ast.err"

conda run -n llvm-dev "$clang" -std=c++17 -fsycl-device-only -fsyntax-only \
  -I "$lesson_dir" \
  -Xclang -ast-dump \
  -Xclang -ast-dump-filter \
  -Xclang "$filter_name" \
  -Xclang -fsycl-int-header="$header_file" \
  -Xclang -fsycl-int-footer="$footer_file" \
  "$source_file" > "$out_dir/$base_name.device.filtered.ast.txt"

conda run -n llvm-dev "$clang" -std=c++17 -fsyntax-only \
  -Xclang -fsycl-is-host \
  -Xclang -ast-dump \
  -I "$lesson_dir" \
  -isystem "$build_include" \
  "$host_wrapper" > "$out_dir/$base_name.host.ast.txt" \
  2> "$out_dir/$base_name.host.ast.err"

conda run -n llvm-dev "$clang" -std=c++17 -fsyntax-only \
  -Xclang -fsycl-is-host \
  -Xclang -ast-dump \
  -Xclang -ast-dump-filter \
  -Xclang "$filter_name" \
  -I "$lesson_dir" \
  -isystem "$build_include" \
  "$host_wrapper" > "$out_dir/$base_name.host.filtered.ast.txt"

conda run -n llvm-dev "$clang" -std=c++17 -fsycl-device-only -S \
  -I "$lesson_dir" \
  -Xclang -emit-llvm \
  -Xclang -disable-llvm-passes \
  -o "$out_dir/$base_name.device.ll" \
  "$source_file" \
  > /dev/null 2> "$out_dir/$base_name.device.ll.err"

conda run -n llvm-dev "$clang" -std=c++17 -S -emit-llvm \
  -Xclang -fsycl-is-host \
  -Xclang -disable-llvm-passes \
  -I "$lesson_dir" \
  -isystem "$build_include" \
  -o "$out_dir/$base_name.host.ll" \
  "$host_wrapper" \
  > /dev/null 2> "$out_dir/$base_name.host.ll.err"

printf 'Generated artifacts in %s\n' "$out_dir"