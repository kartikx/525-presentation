# 525 Presentation

Slides and assets for CS525 Week 2 presentation, authored in Marp Markdown.

## Prerequisites

- Node.js 18+ and npm
- `presentation_v1.md` and assets in `attachments/`

## Install Marp CLI

```bash
npm install --global @marp-team/marp-cli
```

Alternative (without global install):

```bash
npx @marp-team/marp-cli --version
```

## Build Slides

Generate PDF:

```bash
marp presentation_v1.md --pdf --allow-local-files
```

Generate HTML:

```bash
marp presentation_v1.md --html --allow-local-files
```

## VS Code Recommendation

Use the **Marp for VS Code** extension for live preview and faster slide editing.

- Extension ID: `marp-team.marp-vscode`
- Open `presentation_v1.md`
- Run: `Marp: Open Preview` from the command palette
