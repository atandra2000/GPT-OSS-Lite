# GPT-OSS-Lite Static Documentation Site Design

## Goal

Publish the existing GPT-OSS-Lite Markdown documentation as a self-contained static HTML site in `docs_html/`, matching the navigation and interaction quality of the DeepSeek-v3-Lite and LLaMA-3-Lite sites.

## Design

Reuse the proven standard-library Python renderer from the sibling projects. Its renderer converts the project documentation set into HTML pages, preserves headings, tables, code blocks, math placeholders, links, page table-of-contents, sidebar search, theme selection, and previous/next navigation. The GPT-OSS variant will have a GPT-OSS-specific document manifest and portal metadata.

The generated site will contain `index.html`, HTML twins of the project overview and all documents in `docs/`, and one local stylesheet at `docs_html/assets/style.css`. It will not require a package manager, local web server, or third-party build dependency. External CDN URLs are retained only for optional fonts, KaTeX, and syntax highlighting; page content and navigation remain local.

## Validation

The generator must complete with all manifest source paths present. A post-build check will assert that every manifest entry has an HTML twin, `index.html` and the local CSS asset exist, and local HTML links do not target missing generated pages. The existing documentation checks remain the source-of-truth validation for Markdown links and symbol anchors.

## Scope Boundaries

- No edits to the authored Markdown documentation.
- No new documentation framework, dependency, web server, or deployment configuration.
- No remote project files copied into the generated site except optional third-party presentation resources loaded by the browser.
