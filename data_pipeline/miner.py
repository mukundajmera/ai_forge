"""Code Miner - Production-grade AST-based semantic code parser.

This module extracts training data from source code repositories using
Tree-sitter for semantic understanding. It prevents mid-function splits
and preserves code structure for high-quality training data.

Key Features:
    - Multi-language support (Python, JavaScript, TypeScript, Go, Java)
    - Semantic chunking that respects code boundaries  
    - Docstring and comment extraction
    - Function, class, and module-level parsing
    - Quality scoring and filtering
    - Comprehensive error handling

Example:
    >>> from ai_forge.data_pipeline.miner import parse_repository
    >>> blocks = parse_repository("/path/to/project")
    >>> for block in blocks:
    ...     print(f"{block.name}: {block.quality_score():.2f}")
"""

from __future__ import annotations

import logging
import re
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

from data_pipeline.schemas.code_blocks import (
    CodeBlock,
    ExtractionResult,
    LANGUAGE_CONFIGS,
    LanguageConfig,
    MinerConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Language-specific file extension mapping
# =============================================================================

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".h": "c",
}


# =============================================================================
# Tree-sitter Parser Management
# =============================================================================

class ParserManager:
    """Manages Tree-sitter parser instances for different languages.
    
    This class handles lazy initialization of Tree-sitter parsers,
    caching them for reuse across multiple file parsing operations.
    
    Attributes:
        _parsers: Cache of initialized parsers.
        _languages: Cache of loaded language modules.
    """
    
    def __init__(self) -> None:
        """Initialize ParserManager."""
        self._parsers: dict[str, Any] = {}
        self._languages: dict[str, Any] = {}
        self._available: Optional[set[str]] = None
    
    def get_parser(self, language: str) -> Any:
        """Get or create a Tree-sitter parser for a language.
        
        Args:
            language: Language identifier (python, javascript, etc.).
            
        Returns:
            Configured Tree-sitter parser.
            
        Raises:
            ImportError: If Tree-sitter or language grammar not installed.
        """
        if language in self._parsers:
            return self._parsers[language]
        
        try:
            from tree_sitter import Parser
            
            # Get language module
            lang_module = self._get_language(language)
            
            # Create and configure parser
            parser = Parser()
            parser.language = lang_module
            
            self._parsers[language] = parser
            logger.debug(f"Initialized Tree-sitter parser for {language}")
            
            return parser
            
        except ImportError as e:
            raise ImportError(
                f"Tree-sitter parser for {language} not available. "
                f"Install with: pip install tree-sitter tree-sitter-{language}"
            ) from e
    
    def _get_language(self, language: str) -> Any:
        """Get Tree-sitter language module.
        
        Args:
            language: Language identifier.
            
        Returns:
            Tree-sitter Language object.
        """
        if language in self._languages:
            return self._languages[language]
        
        # Map language names to package names
        package_map = {
            "python": "tree_sitter_python",
            "javascript": "tree_sitter_javascript",
            "typescript": "tree_sitter_typescript",
            "go": "tree_sitter_go",
            "java": "tree_sitter_java",
            "rust": "tree_sitter_rust",
            "cpp": "tree_sitter_cpp",
            "c": "tree_sitter_c",
        }
        
        package_name = package_map.get(language)
        if not package_name:
            raise ImportError(f"No Tree-sitter package mapping for language: {language}")
        
        try:
            import importlib
            lang_module = importlib.import_module(package_name)
            
            # Get the language object (usually a function called language())
            if hasattr(lang_module, 'language'):
                lang = lang_module.language()
            else:
                # For older versions, try different patterns
                lang = getattr(lang_module, language.upper(), None)
                if lang is None:
                    raise AttributeError(f"Cannot find language in {package_name}")
            
            self._languages[language] = lang
            return lang
            
        except ImportError:
            raise ImportError(
                f"Tree-sitter language package not installed: {package_name}. "
                f"Install with: pip install {package_name.replace('_', '-')}"
            )
    
    def is_available(self, language: str) -> bool:
        """Check if a language parser is available.
        
        Args:
            language: Language identifier.
            
        Returns:
            True if parser can be initialized.
        """
        try:
            self.get_parser(language)
            return True
        except (ImportError, Exception):
            return False
    
    def available_languages(self) -> set[str]:
        """Get set of available language parsers.
        
        Returns:
            Set of language names with available parsers.
        """
        if self._available is not None:
            return self._available
        
        self._available = set()
        for lang in LANGUAGE_CONFIGS.keys():
            if self.is_available(lang):
                self._available.add(lang)
        
        return self._available


# Global parser manager instance
_parser_manager = ParserManager()


# =============================================================================
# AST Traversal Helpers
# =============================================================================

def _get_node_text(node: Any, source_bytes: bytes) -> str:
    """Extract text content from an AST node.
    
    Args:
        node: Tree-sitter node.
        source_bytes: Original source as bytes.
        
    Returns:
        Text content of the node.
    """
    return source_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='replace')


def _find_docstring_python(node: Any, source_bytes: bytes) -> Optional[str]:
    """Extract Python docstring from function/class.
    
    Args:
        node: Function or class node.
        source_bytes: Source code bytes.
        
    Returns:
        Docstring content or None.
    """
    # Look for expression_statement as first child of body
    for child in node.children:
        if child.type == 'block':
            for stmt in child.children:
                if stmt.type == 'expression_statement':
                    for expr in stmt.children:
                        if expr.type == 'string':
                            text = _get_node_text(expr, source_bytes)
                            # Remove quotes
                            if text.startswith('"""') or text.startswith("'''"):
                                return text[3:-3].strip()
                            elif text.startswith('"') or text.startswith("'"):
                                return text[1:-1].strip()
                    break
                elif stmt.type not in ('comment', 'newline'):
                    break
    return None


def _find_docstring_jsdoc(node: Any, source_bytes: bytes, all_nodes: list) -> Optional[str]:
    """Extract JSDoc comment for function/class.
    
    Looks for a block comment immediately preceding the node.
    
    Args:
        node: Function or class node.
        source_bytes: Source code bytes.
        all_nodes: All nodes at the same level.
        
    Returns:
        JSDoc content or None.
    """
    # Find the index of current node
    node_start_line = node.start_point[0]
    
    # Look for preceding comment
    for sibling in all_nodes:
        if sibling.type == 'comment':
            comment_end_line = sibling.end_point[0]
            # Check if comment ends just before function (within 1 line)
            if comment_end_line == node_start_line - 1 or comment_end_line == node_start_line:
                text = _get_node_text(sibling, source_bytes)
                if text.startswith('/**'):
                    # Parse JSDoc
                    return text[3:-2].strip() if text.endswith('*/') else text[3:].strip()
    
    return None


def _get_function_name(node: Any, source_bytes: bytes, language: str) -> str:
    """Extract function name from AST node.
    
    Args:
        node: Function definition node.
        source_bytes: Source code bytes.
        language: Programming language.
        
    Returns:
        Function name or "<anonymous>".
    """
    # Python: function_definition -> name: identifier
    # JavaScript: function_declaration -> name: identifier
    # Go: function_declaration -> name: identifier
    
    for child in node.children:
        if child.type == 'identifier' or child.type == 'name':
            return _get_node_text(child, source_bytes)
        elif child.type == 'property_identifier':  # JS methods
            return _get_node_text(child, source_bytes)
    
    # Try field access
    try:
        if hasattr(node, 'child_by_field_name'):
            name_node = node.child_by_field_name('name')
            if name_node:
                return _get_node_text(name_node, source_bytes)
    except Exception:
        pass
    
    return "<anonymous>"


def _get_class_name(node: Any, source_bytes: bytes, language: str) -> str:
    """Extract class name from AST node.
    
    Args:
        node: Class definition node.
        source_bytes: Source code bytes.
        language: Programming language.
        
    Returns:
        Class name or "<anonymous>".
    """
    for child in node.children:
        if child.type == 'identifier' or child.type == 'name':
            return _get_node_text(child, source_bytes)
        elif child.type == 'type_identifier':  # Go, Java
            return _get_node_text(child, source_bytes)
    
    # Try field access
    try:
        if hasattr(node, 'child_by_field_name'):
            name_node = node.child_by_field_name('name')
            if name_node:
                return _get_node_text(name_node, source_bytes)
    except Exception:
        pass
    
    return "<anonymous>"


def _extract_dependencies(node: Any, source_bytes: bytes, language: str) -> list[str]:
    """Extract dependencies referenced in a code block.
    
    Args:
        node: AST node to analyze.
        source_bytes: Source code bytes.
        language: Programming language.
        
    Returns:
        List of dependency names (imported modules, called functions).
    """
    deps: set[str] = set()
    source = _get_node_text(node, source_bytes)
    
    if language == "python":
        # Find import statements
        import_pattern = r'(?:from\s+(\w+)|import\s+(\w+))'
        for match in re.finditer(import_pattern, source):
            dep = match.group(1) or match.group(2)
            if dep:
                deps.add(dep)
    
    elif language in ("javascript", "typescript"):
        # Find import/require
        import_pattern = r"(?:import.*from\s+['\"]([^'\"]+)['\"]|require\(['\"]([^'\"]+)['\"]\))"
        for match in re.finditer(import_pattern, source):
            dep = match.group(1) or match.group(2)
            if dep:
                deps.add(dep)
    
    elif language == "go":
        # Find imports
        import_pattern = r'import\s+"([^"]+)"'
        for match in re.finditer(import_pattern, source):
            deps.add(match.group(1))
    
    elif language == "java":
        # Find imports
        import_pattern = r'import\s+([\w.]+);'
        for match in re.finditer(import_pattern, source):
            deps.add(match.group(1))
    
    return list(deps)


# =============================================================================
# Core Extraction Functions
# =============================================================================

def extract_functions(
    code_bytes: bytes,
    language: str,
    include_private: bool = False,
) -> list[Tuple[str, Optional[str], str]]:
    """Parse code using Tree-sitter and extract function definitions.
    
    Walks the AST to find function/method nodes and extracts:
    - Function name
    - Docstring (if present)
    - Full source code
    
    Args:
        code_bytes: Source code as bytes.
        language: Programming language identifier.
        include_private: Whether to include private/underscore functions.
        
    Returns:
        List of (function_name, docstring, full_source_code) tuples.
        
    Example:
        >>> code = b"def hello():\\n    '''Say hello.'''\\n    print('Hello')"
        >>> funcs = extract_functions(code, "python")
        >>> print(funcs[0])  # ('hello', 'Say hello.', 'def hello():...')
    """
    if language not in LANGUAGE_CONFIGS:
        logger.warning(f"Unsupported language: {language}")
        return []
    
    config = LANGUAGE_CONFIGS[language]
    
    try:
        parser = _parser_manager.get_parser(language)
    except ImportError as e:
        logger.warning(f"Parser not available for {language}: {e}")
        return []
    
    try:
        tree = parser.parse(code_bytes)
    except Exception as e:
        logger.error(f"Failed to parse code: {e}")
        return []
    
    results: list[Tuple[str, Optional[str], str]] = []
    
    def walk_tree(node: Any, siblings: list[Any] | None = None) -> None:
        """Recursively walk AST tree."""
        if node.type in config.function_node_types:
            name = _get_function_name(node, code_bytes, language)
            
            # Filter private functions
            if not include_private and name.startswith('_') and not name.startswith('__'):
                return
            
            # Filter test functions
            if name.startswith('test_') or name.endswith('_test'):
                pass  # Include tests by default
            
            # Extract docstring
            if language == "python":
                docstring = _find_docstring_python(node, code_bytes)
            else:
                docstring = _find_docstring_jsdoc(node, code_bytes, siblings or [])
            
            source = _get_node_text(node, code_bytes)
            results.append((name, docstring, source))
        
        # Recurse into children
        children = list(node.children)
        for child in children:
            walk_tree(child, children)
    
    walk_tree(tree.root_node)
    return results


def extract_classes(
    code_bytes: bytes,
    language: str,
) -> list[Tuple[str, Optional[str], list[str]]]:
    """Parse code and extract class definitions with their methods.
    
    Args:
        code_bytes: Source code as bytes.
        language: Programming language identifier.
        
    Returns:
        List of (class_name, docstring, method_names) tuples.
        
    Example:
        >>> code = b"class Calculator:\\n    def add(self, a, b): return a + b"
        >>> classes = extract_classes(code, "python")
        >>> print(classes[0])  # ('Calculator', None, ['add'])
    """
    if language not in LANGUAGE_CONFIGS:
        logger.warning(f"Unsupported language: {language}")
        return []
    
    config = LANGUAGE_CONFIGS[language]
    
    try:
        parser = _parser_manager.get_parser(language)
    except ImportError as e:
        logger.warning(f"Parser not available for {language}: {e}")
        return []
    
    try:
        tree = parser.parse(code_bytes)
    except Exception as e:
        logger.error(f"Failed to parse code: {e}")
        return []
    
    results: list[Tuple[str, Optional[str], list[str]]] = []
    
    def extract_method_names(class_node: Any) -> list[str]:
        """Extract method names from a class node."""
        methods: list[str] = []
        
        def walk(node: Any) -> None:
            if node.type in config.function_node_types:
                name = _get_function_name(node, code_bytes, language)
                if not name.startswith('_') or name.startswith('__'):
                    methods.append(name)
            # Only go one level deep for methods
            if node.type not in config.class_node_types:
                for child in node.children:
                    walk(child)
        
        for child in class_node.children:
            walk(child)
        
        return methods
    
    def walk_tree(node: Any, siblings: list | None = None) -> None:
        if node.type in config.class_node_types:
            name = _get_class_name(node, code_bytes, language)
            
            if language == "python":
                docstring = _find_docstring_python(node, code_bytes)
            else:
                docstring = _find_docstring_jsdoc(node, code_bytes, siblings or [])
            
            methods = extract_method_names(node)
            results.append((name, docstring, methods))
        
        children = list(node.children)
        for child in children:
            walk_tree(child, children)
    
    walk_tree(tree.root_node)
    return results


def extract_module_info(code_bytes: bytes, language: str = "python") -> dict[str, Any]:
    """Extract module-level metadata from source code.
    
    Extracts:
    - Module docstring
    - Imports/dependencies
    - Top-level constants
    - Exported names
    
    Args:
        code_bytes: Source code as bytes.
        language: Programming language (default: python).
        
    Returns:
        Dictionary with module metadata.
        
    Example:
        >>> code = b'''\"\"\"This module does X.\"\"\"\\nimport os\\nVERSION = "1.0"'''
        >>> info = extract_module_info(code)
        >>> print(info["docstring"])  # "This module does X."
    """
    result: dict[str, Any] = {
        "docstring": None,
        "imports": [],
        "constants": [],
        "exports": [],
        "language": language,
    }
    
    source = code_bytes.decode('utf-8', errors='replace')
    
    if language == "python":
        # Module docstring (first string literal)
        docstring_match = re.match(r'^[\s]*(?:["\']{3})(.*?)(?:["\']{3})', source, re.DOTALL)
        if docstring_match:
            result["docstring"] = docstring_match.group(1).strip()
        
        # Imports
        import_pattern = r'^(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))'
        for match in re.finditer(import_pattern, source, re.MULTILINE):
            imp = match.group(1) or match.group(2)
            if imp:
                result["imports"].append(imp)
        
        # Constants (UPPER_CASE = value)
        const_pattern = r'^([A-Z][A-Z0-9_]*)\s*='
        for match in re.finditer(const_pattern, source, re.MULTILINE):
            result["constants"].append(match.group(1))
        
        # Exports (__all__ list)
        all_match = re.search(r'__all__\s*=\s*\[(.*?)\]', source, re.DOTALL)
        if all_match:
            exports = re.findall(r'["\'](\w+)["\']', all_match.group(1))
            result["exports"] = exports
    
    elif language in ("javascript", "typescript"):
        # JSDoc module docstring
        docstring_match = re.match(r'^[\s]*/\*\*(.*?)\*/', source, re.DOTALL)
        if docstring_match:
            result["docstring"] = docstring_match.group(1).strip()
        
        # Imports
        import_pattern = r"(?:import.*from\s+['\"]([^'\"]+)['\"]|require\(['\"]([^'\"]+)['\"]\))"
        for match in re.finditer(import_pattern, source):
            imp = match.group(1) or match.group(2)
            if imp:
                result["imports"].append(imp)
        
        # Exports
        export_pattern = r'export\s+(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)'
        for match in re.finditer(export_pattern, source):
            result["exports"].append(match.group(1))
    
    elif language == "go":
        # Package doc comment
        docstring_match = re.match(r'^[\s]*//\s*(.*?)(?:\n\s*package)', source, re.DOTALL)
        if docstring_match:
            result["docstring"] = docstring_match.group(1).strip()
        
        # Imports
        import_match = re.search(r'import\s+\((.*?)\)', source, re.DOTALL)
        if import_match:
            imports = re.findall(r'"([^"]+)"', import_match.group(1))
            result["imports"] = imports
    
    return result


def parse_repository(
    repo_path: str,
    config: Optional[MinerConfig] = None,
) -> list[CodeBlock]:
    """Parse a repository and extract all semantic code blocks.
    
    Recursively traverses the repository, parsing all supported code files
    and extracting complete syntactic units (functions, classes, modules).
    
    Args:
        repo_path: Path to the repository root.
        config: Mining configuration (optional).
        
    Returns:
        List of CodeBlock objects representing all extracted code units.
        
    Example:
        >>> blocks = parse_repository("/path/to/my/project")
        >>> print(f"Extracted {len(blocks)} code blocks")
        >>> high_quality = [b for b in blocks if b.quality_score() >= 0.7]
    """
    config = config or MinerConfig()
    repo = Path(repo_path)
    
    if not repo.exists():
        raise ValueError(f"Repository path does not exist: {repo_path}")
    
    blocks: list[CodeBlock] = []
    files_processed = 0
    errors: list[str] = []
    
    logger.info(f"Starting repository scan: {repo_path}")
    
    def should_exclude(file_path: Path) -> bool:
        """Check if file matches exclusion patterns."""
        str_path = str(file_path)
        for pattern in config.exclude_patterns:
            if fnmatch(str_path, pattern):
                return True
        return False
    
    def is_test_file(file_path: Path) -> bool:
        """Check if file is a test file."""
        name = file_path.name.lower()
        path_str = str(file_path).lower()
        return (
            name.startswith('test_') or
            name.endswith('_test.py') or
            name.endswith('.test.js') or
            name.endswith('.spec.js') or
            '/tests/' in path_str or
            '/test/' in path_str
        )
    
    # Scan for files
    for file_path in repo.rglob('*'):
        if not file_path.is_file():
            continue
        
        # Get language from extension
        ext = file_path.suffix.lower()
        language = EXTENSION_TO_LANGUAGE.get(ext)
        
        if not language:
            continue
        
        if language not in config.languages:
            continue
        
        if should_exclude(file_path):
            continue
        
        if not config.include_tests and is_test_file(file_path):
            continue
        
        # Parse file
        try:
            # Try multiple encodings
            code_bytes: Optional[bytes] = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    code_bytes = content.encode('utf-8')
                    break
                except UnicodeDecodeError:
                    continue
            
            if code_bytes is None:
                errors.append(f"Encoding error: {file_path}")
                continue
            
            files_processed += 1
            rel_path = str(file_path.relative_to(repo))
            
            # Extract functions
            functions = extract_functions(code_bytes, language, config.include_private)
            for name, docstring, source in functions:
                # Calculate line numbers
                source_start = content.find(source)
                start_line = content[:source_start].count('\n') + 1 if source_start >= 0 else 1
                end_line = start_line + source.count('\n')
                
                block = CodeBlock(
                    path=rel_path,
                    language=language,
                    block_type="function",
                    name=name,
                    docstring=docstring,
                    source_code=source,
                    dependencies=_extract_dependencies(
                        _parser_manager.get_parser(language).parse(code_bytes).root_node,
                        code_bytes,
                        language
                    ) if _parser_manager.is_available(language) else [],
                    metadata={
                        "start_line": start_line,
                        "end_line": end_line,
                        "file_path": str(file_path),
                    }
                )
                
                # Apply quality filters
                if block.token_count < config.min_tokens:
                    continue
                if block.token_count > config.max_tokens:
                    continue
                if config.require_docstring and not block.has_docstring:
                    continue
                if block.quality_score() < config.min_quality_score:
                    continue
                
                blocks.append(block)
            
            # Extract classes
            classes = extract_classes(code_bytes, language)
            for name, docstring, methods in classes:
                # Find class source
                class_pattern = rf'class\s+{re.escape(name)}'
                match = re.search(class_pattern, content)
                if match:
                    # Get class source (naive: until next class or end)
                    start_pos = match.start()
                    # Find end of class (indentation-based for Python)
                    if language == "python":
                        lines = content[start_pos:].split('\n')
                        class_lines = [lines[0]]
                        for line in lines[1:]:
                            if line and not line[0].isspace() and not line.startswith('#'):
                                break
                            class_lines.append(line)
                        source = '\n'.join(class_lines)
                    else:
                        # For braced languages, find matching }
                        brace_count = 0
                        end_pos = start_pos
                        started = False
                        for i, c in enumerate(content[start_pos:]):
                            if c == '{':
                                brace_count += 1
                                started = True
                            elif c == '}':
                                brace_count -= 1
                            if started and brace_count == 0:
                                end_pos = start_pos + i + 1
                                break
                        source = content[start_pos:end_pos]
                    
                    start_line = content[:start_pos].count('\n') + 1
                    end_line = start_line + source.count('\n')
                    
                    block = CodeBlock(
                        path=rel_path,
                        language=language,
                        block_type="class",
                        name=name,
                        docstring=docstring,
                        source_code=source,
                        dependencies=[],
                        metadata={
                            "start_line": start_line,
                            "end_line": end_line,
                            "methods": methods,
                            "method_count": len(methods),
                        }
                    )
                    
                    # Apply filters
                    if block.token_count < config.min_tokens:
                        continue
                    if block.token_count > config.max_tokens:
                        continue
                    if config.require_docstring and not block.has_docstring:
                        continue
                    
                    blocks.append(block)
            
        except Exception as e:
            errors.append(f"Error parsing {file_path}: {e}")
            logger.warning(f"Failed to parse {file_path}: {e}")
    
    logger.info(
        f"Extraction complete: {len(blocks)} blocks from {files_processed} files. "
        f"Errors: {len(errors)}"
    )
    
    return blocks


# =============================================================================
# Convenience Functions
# =============================================================================

def scan_repository_stats(repo_path: str) -> dict[str, Any]:
    """Quick scan to get repository statistics without full parsing.
    
    Args:
        repo_path: Path to repository.
        
    Returns:
        Dictionary with file counts, languages, and estimated LOC.
    """
    repo = Path(repo_path)
    stats: dict[str, Any] = {
        "total_files": 0,
        "by_language": {},
        "total_lines": 0,
        "parseable_files": 0,
    }
    
    for file_path in repo.rglob('*'):
        if not file_path.is_file():
            continue
        
        ext = file_path.suffix.lower()
        language = EXTENSION_TO_LANGUAGE.get(ext)
        
        if language:
            stats["parseable_files"] += 1
            stats["by_language"][language] = stats["by_language"].get(language, 0) + 1
            
            try:
                lines = file_path.read_text(errors='ignore').count('\n')
                stats["total_lines"] += lines
            except Exception:
                pass
        
        stats["total_files"] += 1
    
    return stats


def filter_by_quality(
    blocks: list[CodeBlock],
    min_score: float = 0.6,
) -> list[CodeBlock]:
    """Filter blocks by quality score.
    
    Args:
        blocks: List of code blocks.
        min_score: Minimum quality score (0-1).
        
    Returns:
        Filtered list of high-quality blocks.
    """
    return [b for b in blocks if b.quality_score() >= min_score]


def export_to_json(blocks: list[CodeBlock], output_path: str) -> None:
    """Export code blocks to JSON file.
    
    Args:
        blocks: List of code blocks.
        output_path: Output file path.
    """
    import json
    
    data = [block.to_dict() for block in blocks]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Exported {len(blocks)} blocks to {output_path}")
