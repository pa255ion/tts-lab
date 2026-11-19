{
  description = "tts-lab: a toy text-to-speech lab comparing formant, diphone, unit-selection and HMM synthesis (Zig backend, Qt GUI)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      common = [
        pkgs.zig
        pkgs.ffmpeg-headless # spectrograms: tools/spectrogram.sh
        pkgs.pipewire        # pw-play for `ttslab` playback
        pkgs.curl            # tools/fetch-cmudict.sh
      ];
    in {
      devShells.${system} = {
        # `nix develop`: backend + CLI
        default = pkgs.mkShell {
          packages = common;
        };

        # `nix develop .#gui`: adds Qt 6 + CMake for gui/
        gui = pkgs.mkShell {
          packages = common ++ [
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            pkgs.qt6.qtbase
            pkgs.qt6.qtmultimedia
            pkgs.qt6.wrapQtAppsHook
          ];
        };

        # `nix develop .#eval`: speech recognition for the intelligibility check
        eval = pkgs.mkShell {
          packages = common ++ [
            pkgs.whisper-cpp
            (pkgs.python3.withPackages (p: [ p.numpy ])) # tools/acoustics.py
          ];
        };
      };
    };
}
