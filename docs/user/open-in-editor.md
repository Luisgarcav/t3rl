# Open files in an external editor

t3RL can open a project, file, or exact source location in an editor installed on the same machine as the t3RL server. Use the **Open in** menu in the chat header or beside an openable file reference.

The menu only lists editors detected on that machine. Helix and Neovim are supported as terminal editors:

- Helix: install the `hx` or `helix` command.
- Neovim: install the `nvim` command.

A graphical terminal is also required. On Linux, t3RL detects common terminals such as the system `x-terminal-emulator`, Ghostty, Alacritty, Kitty, WezTerm, Foot, GNOME Console or Terminal, Konsole, and XTerm. macOS uses Terminal, and Windows uses Windows Terminal.

When a file reference includes a line and column, t3RL opens Helix or Neovim at that location. Opening a project starts the editor at the project directory.

## Remote environments

Helix and Neovim are available when the client can ask its environment to launch local processes. They do not currently appear for SSH deep-link mode because these terminal editors do not provide the VS Code-compatible remote URL scheme used by t3RL. In that mode, use a supported graphical editor or open the terminal editor from a shell connected to the remote environment.

If an installed editor does not appear, make sure its command and a supported terminal are available on the server process's `PATH`, then restart t3RL or reconnect after the editor-discovery cache refreshes.
