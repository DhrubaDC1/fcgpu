# fcgpu — FilmCraft GPU Export

Exports [FilmCraft](https://getartcraft.com/) projects on the GPU. FilmCraft's own export composites frames on
the CPU (~3.4 fps for 4K here); fcgpu decodes with VAAPI, grades/reframes/composites with Vulkan (libplacebo)
and encodes with VAAPI — about 3.7x faster on a Radeon RX 5500 XT.

```
fcgpu                                      # window: pick a project, pick an output, Export
fcgpu export PROJECT.fcproj OUT.mp4 [--codec h264|hevc] [--mbps N] [--jobs N] [--check N]
```

## How it stays faithful
FilmCraft itself renders the look once per export: a Hald colour grid (→ 3D LUT per distinct clip Lumetri +
the adjustment-layer grade), grey mattes (→ vignette/letterbox gain map), black/white mattes (→ title matte) and
the audio mix. Geometry (scale/position keyframes) and dip-to-black fades are replayed from the timeline.
`--check N` compares N frames against FilmCraft's own render.

## Supported
One footage track, dip-to-black transitions, linear scale/position keyframes, one adjustment layer (Lumetri +
crop), static titles with cross-dissolve in/out. Anything else stops the export with a list of what's
unsupported. The project needs an empty video track between the footage and the adjustment layer.

## Requires
`filmcraft-cli`, ffmpeg with libplacebo + VAAPI, Python 3 with numpy and tkinter.
