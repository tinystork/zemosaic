# ZeMosaic 4.4.1

- Platform: Windows x64
- Package: Inno Setup installer
- GPU: bundled CuPy CUDA 12.x build
- Fallback: automatic CPU fallback when CUDA/CuPy is unavailable or incompatible

## Notes

- This release is intended for Windows x64 systems.
- NVIDIA GPU acceleration is optional.
- If the GPU runtime cannot be initialized, ZeMosaic falls back to CPU mode automatically.
- The packaged installer should be published as a GitHub Release asset, not committed into Git history.
