# Copyright 2023 Efabless Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
{
  description = "open-source infrastructure for implementing chip design flows";
  
  nixConfig = {
    extra-substituters = [
      "https://openlane.cachix.org"
    ];
    extra-trusted-public-keys = [
      "openlane.cachix.org-1:qqdwh+QMNGmZAuyeQJTH9ErW57OWSvdtuwfBKdS254E="
    ];
  };

  inputs = {
    nix-eda.url = github:efabless/nix-eda/2.1.2;
    libparse.url = github:efabless/libparse-python;
    ioplace-parser.url = github:efabless/ioplace_parser;
    volare.url = github:efabless/volare;
    devshell.url = github:numtide/devshell;
    flake-compat.url = "https://flakehub.com/f/edolstra/flake-compat/1.tar.gz";
  };

  inputs.libparse.inputs.nixpkgs.follows = "nix-eda/nixpkgs";
  inputs.ioplace-parser.inputs.nix-eda.follows = "nix-eda";
  inputs.volare.inputs.nixpkgs.follows = "nix-eda/nixpkgs";
  inputs.devshell.inputs.nixpkgs.follows = "nix-eda/nixpkgs";

  outputs = {
    self,
    nix-eda,
    libparse,
    ioplace-parser,
    volare,
    devshell,
    ...
  }: let
    nixpkgs = nix-eda.inputs.nixpkgs;
    lib = nixpkgs.lib;
  in {
    # Common
    overlays = {
      default = lib.composeManyExtensions [
        (import ./nix/overlay.nix)
        (nix-eda.flakesToOverlay [libparse ioplace-parser volare])
        (pkgs': pkgs: {
          yosys-sby = (pkgs.yosys-sby.override { sha256 = "sha256-Il2pXw2doaoZrVme2p0dSUUa8dCQtJJrmYitn1MkTD4="; });
          klayout = (pkgs.klayout.overrideAttrs(old: {
            configurePhase = builtins.replaceStrings ["-without-qtbinding"] ["-with-qtbinding"] old.configurePhase;
          }));
          yosys = pkgs.yosys.overrideAttrs(old: {
            patches = old.patches ++ [
              ./nix/patches/yosys/async_rules.patch
            ];
          });
        })
        (
          pkgs': pkgs: let
            callPackage = lib.callPackageWith pkgs';
          in {
            colab-env = callPackage ./nix/colab-env.nix {};
            opensta = callPackage ./nix/opensta.nix {};
            openroad-abc = callPackage ./nix/openroad-abc.nix {};
            openroad = callPackage ./nix/openroad.nix {};
          }
        )
        (
          nix-eda.composePythonOverlay (pkgs': pkgs: pypkgs': pypkgs: let
            callPythonPackage = lib.callPackageWith (pkgs' // pkgs'.python3.pkgs);
          in {
            mdformat = pypkgs.mdformat.overridePythonAttrs (old: {
              patches = [
                ./nix/patches/mdformat/donns_tweaks.patch
              ];
              doCheck = false;
            });
            sphinx-tippy = callPythonPackage ./nix/sphinx-tippy.nix {};
            sphinx-subfigure = callPythonPackage ./nix/sphinx-subfigure.nix {};
            yamlcore = callPythonPackage ./nix/yamlcore.nix {};

            # ---
            openlane = callPythonPackage ./default.nix {};
          })
        )
        (pkgs': pkgs: let
          callPackage = lib.callPackageWith pkgs';
        in
          {}
          // lib.optionalAttrs pkgs.stdenv.isLinux {
            openlane-docker = callPackage ./nix/docker.nix {
              createDockerImage = nix-eda.createDockerImage;
              openlane = pkgs'.python3.pkgs.openlane;
            };
          })
      ];
    };

    # Helper functions
    createOpenLaneShell = import ./nix/create-shell.nix;

    # Packages
    legacyPackages = nix-eda.forAllSystems (
      system:
        import nix-eda.inputs.nixpkgs {
          inherit system;
          overlays = [devshell.overlays.default nix-eda.overlays.default self.overlays.default];
        }
    );

    packages = nix-eda.forAllSystems (
      system: let
        pkgs = (self.legacyPackages."${system}");
        in {
          inherit (pkgs) colab-env opensta openroad-abc openroad;
          inherit (pkgs.python3.pkgs) openlane;
          default = pkgs.python3.pkgs.openlane;
        }
        // lib.optionalAttrs pkgs.stdenv.isLinux {
          inherit (pkgs) openlane-docker;
        }
    );

    # devshells

    devShells = nix-eda.forAllSystems (
      system: let
        pkgs = self.legacyPackages."${system}";
        callPackage = lib.callPackageWith pkgs;
      in {
        # These devShells are rather unorthodox for Nix devShells in that they
        # include the package itself. For a proper devShell, try .#dev.
        
        # --- Use the existing createOpenLaneShell but with a startup script ---
        default = callPackage (self.createOpenLaneShell {
            extra-packages = with pkgs; [
              gcc
              cmake
              ninja
              gnumake
              patchelf
            ];
            extra-python-packages = with pkgs.python3.pkgs; [
              numpy
              torch
              scipy
              pandas
              matplotlib
              seaborn
              scikit-learn
              jupyter
              ipython
              ipywidgets
              pip
              setuptools
              wheel
              virtualenv
            ];
        }) {};
        # ---------------------------------------------------------

        notebook = callPackage (self.createOpenLaneShell {
          extra-packages = with pkgs; [
            jupyter
          ];
        }) {};
        # Normal devShells
        dev = callPackage (self.createOpenLaneShell {
          extra-packages = with pkgs; [
            jdupes
            alejandra
          ];
          extra-python-packages = with pkgs.python3.pkgs; [
            pyfakefs
            pytest
            pytest-xdist
            pytest-cov
            pillow
            mdformat
            black
            ipython
            tokenize-rt
            flake8
            mypy
            types-deprecated
            types-pyyaml
            types-psutil
            lxml-stubs
          ];
          include-openlane = false;
        }) {};
        docs = callPackage (self.createOpenLaneShell {
          extra-packages = with pkgs; [
            jdupes
            alejandra
            imagemagick
            nodejs.pkgs.nodemon
          ];
          extra-python-packages = with pkgs.python3.pkgs; [
            pyfakefs
            pytest
            pytest-xdist
            pillow
            mdformat
            furo
            docutils
            sphinx
            sphinx-autobuild
            sphinx-autodoc-typehints
            sphinx-design
            myst-parser
            docstring-parser
            sphinx-copybutton
            sphinxcontrib-spelling
            sphinxcontrib-bibtex
            sphinx-tippy
            sphinx-subfigure
          ];
          include-openlane = false;
        }) {};
        
        # --- ADD A NEW SHELL SPECIFICALLY FOR PYTORCH GEOMETRIC ---
        pytorch-geometric = pkgs.devshell.mkShell {
          name = "openlane-pytorch-geometric";
          
          packages = with pkgs; [
            # Core OpenLane
            python3.pkgs.openlane
            # Build tools
            gcc
            cmake
            ninja
            gnumake
            patchelf
            # Python and ML
            python3
            python3Packages.numpy
            python3Packages.torch
            python3Packages.scipy
            python3Packages.pandas
            python3Packages.matplotlib
            python3Packages.seaborn
            python3Packages.scikit-learn
            python3Packages.jupyter
            python3Packages.ipython
            python3Packages.ipywidgets
            python3Packages.pip
            python3Packages.setuptools
            python3Packages.wheel
            python3Packages.virtualenv
            # EDA tools
            verilog
            gtkwave
            # Utilities
            git
            curl
            wget
            vim
            bat
            ripgrep
            fd
          ];
          
          env = [
            {
              name = "PYTHONPATH";
              value = "$PYTHONPATH:${pkgs.python3.pkgs.openlane}/${pkgs.python3.sitePackages}";
            }
          ];
          
          commands = [
            {
              name = "openlane";
              help = "OpenLane CLI";
              command = "python -m openlane $@";
            }
            {
              name = "setup-pytorch-geometric";
              help = "Setup PyTorch Geometric in a virtual environment";
              command = ''
                if [ ! -d "$PWD/.venv" ]; then
                  echo "Creating virtual environment..."
                  python -m venv .venv
                fi
                echo "Activating virtual environment..."
                source .venv/bin/activate
                echo "Installing PyTorch Geometric..."
                pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
                echo "Done! Virtual environment is at $PWD/.venv"
                echo "To activate: source .venv/bin/activate"
              '';
            }
          ];
          
          # Devshell uses startup instead of shellHook
          startup = {
            welcome.text = ''
              === OpenLane with PyTorch Geometric Development Shell ===
              
              Available commands:
                • setup-pytorch-geometric - Install PyTorch Geometric in a virtual environment
                • openlane - Run OpenLane CLI
                
              To install PyTorch Geometric:
                1. Run: setup-pytorch-geometric
                2. Activate: source .venv/bin/activate
                3. Test: python -c "import torch_geometric; print('Success!')"
                
              Or manually:
                1. python -m venv .venv
                2. source .venv/bin/activate  
                3. pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
            '';
          };
        };
      }
    );
  };
}