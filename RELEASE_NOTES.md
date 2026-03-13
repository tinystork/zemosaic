# ZeMosaic 4.4.1

## Qt-only / Tk retirement update

- Qt (PySide6) is now the only official frontend runtime path.
- Official startup no longer falls back to Tk (`--tk-gui` unsupported on official path).
- `zemosaic_config` migration normalizes legacy `preferred_gui_backend=tk` to `qt` and neutralizes obsolete backend-selection state.
- `lecropper` remains an annex/standalone legacy tool and is decoupled from official runtime/headless validated paths.

## Compatibility / unsupported legacy

- Legacy Tk frontend is not an official runtime path.
- Full repo-wide removal of all Tk annex tools is out-of-scope for this release line.
