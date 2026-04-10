#!/usr/bin/env python3
"""
Generate a .clangd configuration for JUCE that lets clangd parse any file in
a module — sub-header, .cpp fragment, master header, or module CU — with the
exact context the real build would have at that point.

Usage:
    python3 generate_clangd_juce_context.py [modules_dir]

modules_dir defaults to ./modules; output_dir defaults to its parent.

The .clangd file embeds absolute paths and is host-specific — gitignore it.
The context headers under .clangd-juce/ reference module files via paths
relative to the context directory and pull base + Windows defines from a
shared juce_clangd_defines.h (Windows defines sit behind `#ifdef _WIN32`),
so the emitted context tree works under clang, clang++, and clang-cl
without a driver switch.

Design: see README.md.
"""

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

__version__ = "0.4.0"

## regexes

# Inter-module dep: #include <juce_core/juce_core.h>
MODULE_DEP_INCLUDE_RE = re.compile(
    r'\s*#\s*include\s*[<"](juce_\w+)/\1\.h[>"]'
)

# Quoted include line — captures indent, path, base (stem), and extension.
# Headers (.h/.hpp/.inl) are sub-headers of the master; implementations
# (.cpp/.mm) are CU fragments. Tolerates a trailing `// ...` comment,
# which strip_block_comments_and_errors deliberately leaves intact.
QUOTED_INCLUDE_RE = re.compile(
    r'^(?P<indent>\s*)#\s*include\s*'
    r'"(?P<path>[^"]*?(?P<base>[A-Za-z_]\w*)'
    r'\.(?P<ext>h|hpp|inl|mm|cpp))"'
    r'\s*(?://[^\n]*)?\s*$'
)

ERROR_DIRECTIVE_RE = re.compile(r'^\s*#\s*error\b')

# JUCE root CMakeLists.txt declares: project(JUCE VERSION X.Y.Z ...)
JUCE_VERSION_RE = re.compile(
    r'project\s*\(\s*JUCE\s+VERSION\s+([\d.]+)', re.IGNORECASE
)

# Matches (in order): a double-quoted string, a single-quoted char/string,
# a `//` line comment, or a `/* ... */` block comment. Strings and line
# comments are kept verbatim so a `/*` inside one isn't misread as an
# opener; only block comments are dropped.
TOKEN_RE = re.compile(
    r'"(?:\\.|[^"\\])*"'
    r"|'(?:\\.|[^'\\])*'"
    r"|//[^\n]*"
    r"|/\*[\s\S]*?\*/"
)

# These look like block comments but must survive stripping —
# emit_header_walk keys off them.
COND_MARKERS = frozenset({"/** @cond */", "/** @endcond */"})

HEADER_EXTS = ("h", "hpp", "inl")
IMPL_EXTS = ("cpp", "mm")

DEFINES_HEADER_NAME = "juce_clangd_defines.h"


def as_posix(path: str) -> str:
    """Posix-style form of a filesystem path (drive letters preserved)."""
    return Path(path).as_posix()


@dataclass
class Module:
    """Everything build_context and the .clangd Layer 2/3 emitters need to
    know about one JUCE module."""
    name: str
    dir_abs: str                 # posix absolute path
    header_path: str             # master header, for open()
    cu_rel: str | None           # "<name>.cpp" or "<name>.mm", or None
    cu_path: str | None          # absolute CU path, for open()
    all_deps: list[str]


def strip_block_comments_and_errors(lines: list[str]) -> list[str]:
    """Drop /* ... */ block comments (replaced by a single space per C++
    lex rules) and lines whose first non-whitespace is #error. Preserves
    /** @cond */ / /** @endcond */ markers verbatim. String literals and
    // line comments pass through untouched."""
    def replace(m: re.Match) -> str:
        s = m.group(0)
        if s.startswith("/*") and s not in COND_MARKERS:
            return " "
        return s

    stripped = TOKEN_RE.sub(replace, "".join(lines))
    return [line for line in stripped.splitlines(keepends=True)
            if line.strip() != "" and not ERROR_DIRECTIVE_RE.match(line)]

## module discovery

def read_juce_version(modules_dir: str) -> str | None:
    """JUCE version from the CMakeLists.txt next to modules_dir, or None
    if the file is missing or doesn't declare `project(JUCE VERSION ...)`."""
    cmake_path = Path(modules_dir).parent / "CMakeLists.txt"
    try:
        text = cmake_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = JUCE_VERSION_RE.search(text)
    return m.group(1) if m else None


def find_modules(modules_dir: str) -> list[str]:
    modules = []
    for entry in sorted(os.listdir(modules_dir)):
        module_dir = os.path.join(modules_dir, entry)
        module_header = os.path.join(module_dir, f"{entry}.h")
        if os.path.isdir(module_dir) and os.path.isfile(module_header):
            modules.append(entry)
    return modules


def parse_module_dependencies(modules_dir: str, module_name: str) -> list[str]:
    """Direct dependency modules — scan every source file in the module for
    inter-module `#include <juce_X/juce_X.h>` lines. JUCE convention puts
    these in the master header, but modules like juce_audio_plugin_client
    and juce_audio_processors declare extra deps in CU fragments or helper
    sub-headers, so the whole tree must be scanned."""
    module_dir = os.path.join(modules_dir, module_name)
    exts = tuple(f".{e}" for e in HEADER_EXTS + IMPL_EXTS)
    deps: list[str] = []
    for rel in find_module_files(module_dir, exts):
        path = os.path.join(module_dir, rel)
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = MODULE_DEP_INCLUDE_RE.match(line)
                if m:
                    dep = m.group(1)
                    if dep != module_name and dep not in deps:
                        deps.append(dep)
    return deps


def get_all_transitive_deps(
    module: str, dep_map: dict[str, list[str]], visited: set | None = None
) -> list[str]:
    if visited is None:
        visited = set()
    result = []
    for dep in dep_map.get(module, []):
        if dep not in visited:
            visited.add(dep)
            result.extend(get_all_transitive_deps(dep, dep_map, visited))
            result.append(dep)
    return result


def find_module_files(module_dir: str, exts: tuple[str, ...],
                      exclude: str | None = None) -> list[str]:
    """Collect relative paths of files matching exts under module_dir."""
    result = []
    for root, dirs, files in os.walk(module_dir):
        dirs.sort()
        for f in sorted(files):
            if not f.endswith(exts):
                continue
            rel = os.path.relpath(os.path.join(root, f), module_dir)
            if exclude and rel == exclude:
                continue
            result.append(rel)
    return result

## context generation

def guarded_include(indent: str, stop_at: str, include_path: str) -> list[str]:
    """Stop-at-guarded include, gated on JUCE_CLANGD_STOPPED."""
    return [
        f"{indent}#ifndef JUCE_CLANGD_STOPPED\n",
        f"{indent} #ifdef {stop_at}\n",
        f"{indent}  #define JUCE_CLANGD_STOPPED 1\n",
        f"{indent} #else\n",
        f'{indent}  #include "{include_path}"\n',
        f"{indent} #endif\n",
        f"{indent}#endif\n",
    ]


def emit_define(name: str, value: str | None = "1", indent: str = "") -> str:
    """Idempotent `#define NAME [VALUE]` line. value=None emits a bare define."""
    tail = f" {value}" if value is not None else ""
    return (
        f"{indent}#ifndef {name}\n"
        f"{indent} #define {name}{tail}\n"
        f"{indent}#endif\n"
    )


def rel_include(target_abs: str, context_dir: str) -> str:
    """Posix-style path from `context_dir` to `target_abs`. Falls back to
    absolute form when relpath is impossible (e.g. different Windows drives)."""
    try:
        return as_posix(os.path.relpath(target_abs, context_dir))
    except ValueError:
        return as_posix(os.path.abspath(target_abs))


def emit_header_walk(header_lines: list[str], module: "Module",
                      context_dir: str) -> list[str]:
    """Rewrite the master header: each intra-module juce_-prefixed sub-header
    include becomes a stop-at-guarded include (STOP_AT_<base>_<ext>), with
    the path resolved relative to context_dir. Doxygen @cond/@endcond
    regions wrap in JUCE_CLANGD_PARSE so the link-check sentinel doesn't
    fire. All other lines pass through."""
    out: list[str] = []
    in_cond = False
    for line in header_lines:
        if "/** @cond */" in line and not in_cond:
            out.append("#ifndef JUCE_CLANGD_PARSE\n")
            out.append(line)
            in_cond = True
            continue
        if "/** @endcond */" in line and in_cond:
            out.append(line)
            out.append("#endif // !JUCE_CLANGD_PARSE\n")
            in_cond = False
            continue

        m = QUOTED_INCLUDE_RE.match(line)
        if m and m.group("ext") in HEADER_EXTS:
            base = m.group("base")
            if base.startswith("juce_") and base != module.name:
                target = os.path.join(module.dir_abs, m.group("path"))
                out.extend(guarded_include(
                    m.group("indent"),
                    f"JUCE_CLANGD_STOP_AT_{base}_{m.group('ext')}",
                    rel_include(target, context_dir),
                ))
                continue

        out.append(line)

    if in_cond:
        raise RuntimeError(
            f"{module.header_path}: unbalanced Doxygen /** @cond */ "
            f"without matching /** @endcond */"
        )
    return out


def build_context(module: "Module", context_dir: str) -> str:
    """Generate <module>_context.h — the unified force-include for every file
    in the module. Pulls base and Windows defines from the shared
    juce_clangd_defines.h, emits per-module MODULE_AVAILABLE_* macros, then
    walks the module CU (or master header if header-only), rewriting
    `#include "<module>.h"` into an inline sub-header walk, stripping other
    intra-module header re-includes, and turning .cpp/.mm fragments into
    stop-at-guarded includes. A single JUCE_CLANGD_STOPPED flag plus
    per-file STOP_AT_<base>_<ext> (or STOP_AT_FRAGMENTS for the CU itself)
    halts the walk just before the file clangd is parsing. All include
    paths are written relative to context_dir."""
    with open(module.header_path, encoding="utf-8", errors="replace") as f:
        header_lines = strip_block_comments_and_errors(f.readlines())

    has_cu = module.cu_path is not None and os.path.isfile(module.cu_path)
    cu_lines: list[str] = []
    if has_cu:
        with open(module.cu_path, encoding="utf-8", errors="replace") as f:
            cu_lines = strip_block_comments_and_errors(f.readlines())

    out: list[str] = [
        "// AUTO-GENERATED by generate_clangd_juce_context.py. DO NOT EDIT.\n",
        f"// Unified clangd parse context for JUCE module: {module.name}\n",
        "\n",
        "#pragma once\n\n",
        f'#include "{DEFINES_HEADER_NAME}"\n\n',
        "// Module availability.\n",
    ]
    for m in [*module.all_deps, module.name]:
        out.append(emit_define(f"JUCE_MODULE_AVAILABLE_{m}"))
    out.append("\n")

    if not has_cu:
        out.append("// --- sub-header walk (rewritten master header) ---\n")
        out.extend(emit_header_walk(header_lines, module, context_dir))
        return "".join(out)

    out.append("// --- unified walk (rewritten module CU) ---\n")
    fragment_sentinel_emitted = False
    for line in cu_lines:
        m = QUOTED_INCLUDE_RE.match(line)
        if m:
            base = m.group("base")
            ext = m.group("ext")
            if ext == "h" and base == module.name:
                out.extend(emit_header_walk(header_lines, module, context_dir))
                continue
            if ext in HEADER_EXTS:
                continue
            if ext in IMPL_EXTS:
                if not fragment_sentinel_emitted:
                    out.append("#ifdef JUCE_CLANGD_STOP_AT_FRAGMENTS\n")
                    out.append(" #define JUCE_CLANGD_STOPPED 1\n")
                    out.append("#endif\n")
                    fragment_sentinel_emitted = True
                target = os.path.join(module.dir_abs, m.group("path"))
                out.extend(guarded_include(
                    m.group("indent"),
                    f"JUCE_CLANGD_STOP_AT_{base}_{ext}",
                    rel_include(target, context_dir),
                ))
                continue
        out.append(line)

    return "".join(out)


def write_defines_header(context_dir: str, juce_version: str | None) -> None:
    """Emit the shared defines header every <module>_context.h includes.
    Base defines are unconditional; Windows/MSVC defines (normally set by
    the real compile command) are gated on _WIN32 so one file works under
    clang, clang++, and clang-cl."""
    juce_tag = juce_version if juce_version else "unknown"
    lines: list[str] = [
        f"// AUTO-GENERATED by generate_clangd_juce_context.py v{__version__}. DO NOT EDIT.\n",
        f"// JUCE version: {juce_tag}\n",
        "// Shared preprocessor defines for clangd JUCE context headers.\n",
        "\n",
        "#pragma once\n\n",
        "// Base defines (all platforms).\n",
    ]
    for name, value in [
        ("DEBUG", "1"),
        ("JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED", "1"),
        ("JUCE_CLANGD_PARSE", "1"),
    ]:
        lines.append(emit_define(name, value))

    lines += [
        "\n",
        "// Windows / MSVC defines — normally set on the real command line.\n",
        "#ifdef _WIN32\n",
    ]
    for name, value in [
        ("_DEBUG", "1"),
        ("WIN32", None), ("_WINDOWS", None),
        ("NOMINMAX", None), ("_CRT_SECURE_NO_WARNINGS", None),
        ("UNICODE", None), ("_UNICODE", None),
    ]:
        lines.append(emit_define(name, value, indent=" "))
    lines.append("#endif\n")

    path = os.path.join(context_dir, DEFINES_HEADER_NAME)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)


## flag helpers (GNU-style — clang-cl accepts these alongside its MSVC forms)

def base_flags() -> list[str]:
    """Compiler flags that can't live in a header.
    -fno-modules: prevents implicit Xcode/libc++ module maps causing ODR errors.
    -x objective-c++ (macOS): juce_core.h pulls in Foundation headers."""
    flags = ["-std=c++20", "-Wno-pragma-once-outside-header", "-fno-modules"]
    if sys.platform == "darwin":
        flags += ["-x", "objective-c++"]
    return flags


def include_flag(header_path: str) -> list[str]:
    return ["-include", header_path]


def include_path_flag(dir_path: str) -> str:
    return f"-I{dir_path}"


def define_flag(name: str, value: str = "1") -> str:
    return f"-D{name}={value}"


## .clangd generation

def yaml_block(path_pattern: str | None, flags: list[str],
               remove: list[str] | None = None) -> str:
    """Emit one .clangd YAML block. path_pattern=None → unconditional (no If:)."""
    lines: list[str] = []
    if path_pattern is not None:
        lines.extend(["If:", f"  PathMatch: {path_pattern}"])
    lines.append("CompileFlags:")

    def emit(key: str, items: list[str]) -> None:
        if not items:
            return
        lines.append(f"  {key}:")
        for x in items:
            q = any(c in x for c in ' :#"\'')
            lines.append(f'    - "{x}"' if q else f"    - {x}")

    emit("Remove", remove or [])
    emit("Add", flags)
    return "\n".join(lines)


def generate(modules_dir: str, output_dir: str,
             context_subdir: str = ".clangd-juce") -> None:
    modules_dir = os.path.abspath(modules_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    clangd_path = os.path.join(output_dir, ".clangd")
    context_dir = os.path.join(output_dir, context_subdir)

    if os.path.isdir(context_dir):
        shutil.rmtree(context_dir)
    os.makedirs(context_dir)

    all_modules = find_modules(modules_dir)
    if not all_modules:
        print(f"No JUCE modules found in {modules_dir}", file=sys.stderr)
        sys.exit(1)

    juce_version = read_juce_version(modules_dir)

    print(f"Modules:      {modules_dir}")
    print(f"Output:       {output_dir}")
    print(f"Context dir:  {context_dir}")
    print(f"JUCE version: {juce_version if juce_version else 'unknown'}")
    print(f"Modules found ({len(all_modules)}): {', '.join(all_modules)}")
    print()

    write_defines_header(context_dir, juce_version)

    print("Scanning module dependencies...")
    dep_map: dict[str, list[str]] = {}
    for name in all_modules:
        print(f"  {name}", flush=True)
        dep_map[name] = parse_module_dependencies(modules_dir, name)
    print()

    print("Generating context headers...")

    # Layer 1: global
    blocks: list[str] = [yaml_block(
        None,
        [*base_flags(), include_path_flag(as_posix(modules_dir))],
        remove=["-include"],
    )]

    for name in all_modules:
        module_dir = os.path.join(modules_dir, name)

        # CU is conventionally <module>.cpp; fall back to <module>.mm for the
        # (rare) Obj-C++-only module.
        cu_rel = None
        cu_path = None
        for ext in ("cpp", "mm"):
            candidate = os.path.join(module_dir, f"{name}.{ext}")
            if os.path.isfile(candidate):
                cu_path = candidate
                cu_rel = f"{name}.{ext}"
                break

        module = Module(
            name=name,
            dir_abs=as_posix(os.path.abspath(module_dir)),
            header_path=os.path.join(module_dir, f"{name}.h"),
            cu_rel=cu_rel,
            cu_path=cu_path,
            all_deps=get_all_transitive_deps(name, dep_map),
        )

        context_text = build_context(module, context_dir)
        context_path = os.path.join(context_dir, f"{name}_context.h")
        with open(context_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(context_text)
        force_include = as_posix(os.path.abspath(context_path))

        files = find_module_files(
            module_dir,
            tuple(f".{e}" for e in HEADER_EXTS + IMPL_EXTS),
            exclude=f"{name}.h",
        )
        cu_note = "" if module.cu_rel else " (header-only)"
        print(f"  {name}: context{cu_note}, {len(files)} files")

        # Layer 2: per-module umbrella — force-include the context, -I module dir.
        module_re = re.escape(name)
        blocks.append(yaml_block(
            f".*{module_re}/.*",
            [*include_flag(force_include), include_path_flag(module.dir_abs)],
        ))

        # Layer 3: one stop-at override per file in the module. When clangd
        # parses file X, the walk in the context stops before X so X can be
        # parsed fresh. The module CU is special: its stop is the group-level
        # FRAGMENTS sentinel, which halts before *any* fragment runs.
        for rel in files:
            base, dot_ext = os.path.splitext(os.path.basename(rel))
            ext = dot_ext.lstrip(".")
            if rel == module.cu_rel:
                macro = "JUCE_CLANGD_STOP_AT_FRAGMENTS"
            elif ext in HEADER_EXTS and not base.startswith("juce_"):
                continue  # not in the master-header walk, nothing to stop at
            else:
                macro = f"JUCE_CLANGD_STOP_AT_{base}_{ext}"
            blocks.append(yaml_block(
                f".*{module_re}/{re.escape(as_posix(rel))}",
                [define_flag(macro)],
            ))

    content = "\n\n---\n\n".join(blocks) + "\n"
    with open(clangd_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)

    print()
    print(f"Wrote {clangd_path} ({len(blocks)} blocks).")
    print(f"Wrote context headers + {DEFINES_HEADER_NAME} under {context_dir}/")
    print()
    print("Add to .gitignore:")
    print("    .clangd")
    print(f"    {context_subdir}/")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate .clangd for JUCE modules with one context "
                    "header per module.",
    )
    p.add_argument(
        "modules_dir", nargs="?", default="./modules",
        help="Path to JUCE/modules (default: ./modules next to this script)",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Where to write .clangd and the context dir "
             "(default: parent of modules_dir)",
    )
    p.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
        help="show the version"
    )
    args = p.parse_args()

    if not os.path.isdir(args.modules_dir):
        print(f"Error: {args.modules_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.modules_dir))

    generate(args.modules_dir, args.output_dir)


if __name__ == "__main__":
    main()
