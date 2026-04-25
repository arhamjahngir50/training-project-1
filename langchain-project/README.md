# 🤖 Linux Computer Control Agent

A natural language Linux agent powered by **Gemini 2.5 Flash** + **LangChain**. Just type what you want — it controls your computer.

---

## 🚀 Setup

```bash
pip install langchain langchain-google-genai psutil python-dotenv
```

Create a `.env` file:

```
GEMINI_API_KEY=your_key_here
```

Run the agent:

```bash
python computer_agent.py
```

---

## 🖥️ Built-in Session Commands

| Command     | Description                        |
|-------------|------------------------------------|
| `history`   | Show all past turns this session   |
| `clear`     | Wipe conversation memory           |
| `exit`      | Quit the agent                     |

---

## 📁 Files & Folders

```
"Create a folder called Projects on my Desktop"
"Make a nested folder ~/work/2025/january"
"Delete the folder ~/old-backup"
"Remove the file notes.txt from my Desktop"
"List all files in my Downloads folder"
"What's inside ~/Documents?"
"Read the file ~/notes.txt"
"Show me what's inside config.json"
"Write a README.md to my Desktop"
"Create a file called todo.txt with my task list"
"Add a new line to ~/todo.txt saying buy milk"
"Append today's date to my log file"
"Copy my resume.pdf to the Desktop"
"Duplicate the folder ~/templates to ~/backup"
"Move report.pdf from Downloads to Documents"
"Rename hello.py to main.py"
"Find all PDF files in my home folder"
"Search for all .log files in /var/log"
"Find every .mp4 file on my computer"
"Find all files named config.json"
```

---

## 💻 Coding & Scripts

```
"Write a Python hello world script to ~/test.py and run it"
"Create a bash script that backs up my Desktop to ~/backup"
"Write a Python script that reads a CSV and prints rows"
"Run the script at ~/scripts/backup.sh"
"Run my Python file ~/app.py"
"Compile and run hello.c"
"Run npm install in ~/myproject"
"Execute make in ~/myproject"
"Check Python version"
"Check Node.js version"
"Create a virtual environment in ~/myproject"
"Activate venv and install flask"
"Install the requests package"
"Install numpy scipy matplotlib with pip"
"List all installed Python packages"
"Show me the git log of ~/myproject"
"Initialize a git repo in ~/myproject"
"Run git status in my project folder"
```

---

## 📊 System Monitoring

```
"What are my system specs?"
"Show me CPU and RAM usage"
"How much disk space is left?"
"How long has my computer been on?"
"List all running processes"
"Show processes with 'python' in the name"
"What processes are using the most RAM?"
"Is chrome running?"
"Kill firefox"
"Stop the process with PID 4321"
"Close all chrome processes"
"What time is it?"
"What's today's date?"
"Check CPU temperature"
"Show me live CPU usage every second"
"Who is logged in right now?"
```

---

## 📦 Package Management

```
"Install git with apt"
"Install vlc media player"
"Update all system packages"
"Upgrade the system"
"Remove a package called gimp"
"Search for a package called ffmpeg"
"List all installed apt packages"
"Install snap and then install spotify"
"Install docker"
"Check if curl is installed"
```

---

## 🌐 Network & Internet

```
"Am I connected to the internet?"
"What is my local IP address?"
"What is my public IP address?"
"Show all network interfaces"
"Ping google.com 5 times"
"Download a file from a URL"
"Check what ports are open"
"Show active network connections"
"Test speed of my internet"
"Flush DNS cache"
"Show my Wi-Fi name and signal strength"
"Traceroute to github.com"
```

---

## 🚀 Applications

```
"Open Firefox"
"Launch VS Code"
"Open the file manager"
"Open terminal"
"Start VLC"
"Open report.pdf"
"Open the image ~/photos/trip.jpg"
"Open my project in VS Code"
"Close firefox"
"Restart the audio service"
```

---

## 🗜️ File Compression & Archiving

```
"Compress the folder ~/photos into photos.tar.gz"
"Extract archive.tar.gz to ~/extracted"
"Zip the folder ~/documents"
"Unzip myfile.zip to ~/Downloads"
"Create a tar backup of my home folder"
"List contents of archive.tar.gz without extracting"
```

---

## 📝 Text & File Processing

```
"Search for the word 'error' in all .log files"
"Count lines in a file"
"Replace all occurrences of 'foo' with 'bar' in a file"
"Sort the lines in ~/list.txt alphabetically"
"Show the last 50 lines of a log file"
"Show lines containing 'ERROR' in syslog"
"Compare two files and show differences"
"Convert a file from Windows line endings to Unix"
"Show word count of a document"
"Merge two text files into one"
```

---

## 🔒 Permissions & Users

```
"Make a script executable"
"Change permissions of ~/myfile.sh to 755"
"Who owns this file?"
"Change owner of a file to my user"
"List all users on this system"
"Show my username"
"Show sudo privileges"
```

---

## 🎬 Media & Display

```
"Take a screenshot and save to Desktop"
"Set volume to 70%"
"Mute the system — set volume to 0"
"Set volume to max"
"Get current clipboard content"
"Copy this text to clipboard: hello world"
"Get screen resolution"
"List connected monitors"
"Convert image.png to image.jpg"
"Resize an image to 800x600"
"Get duration of a video file"
```

---

## 🧠 Conversation Memory

The agent remembers the last **20 turns** of your conversation. You can use context from earlier messages without repeating yourself.

```
"Now delete that folder"
"Run the script I just wrote"
"What did I ask you to create?"
"Add a comment to the file we just made"
"Do the same thing but in ~/Documents"
"Now open it"
```

---

## ⚡ How It Works

The agent uses **two layers** to handle commands:

| Layer | What it does |
|-------|-------------|
| **Dedicated tools** | Fast, structured actions — create folder, list files, kill process, screenshot, volume, etc. |
| **`run_shell_command`** | Catch-all — executes any raw Linux shell command. If Linux can do it from the terminal, the agent can do it. |

> **The command list above is not exhaustive.** Since the agent can run any shell command, it can do anything your Linux terminal can — the possibilities are effectively unlimited.

---

## 🛠️ Tools Reference

| Tool | Description |
|------|-------------|
| `create_folder` | Create folders with any path |
| `delete_file_or_folder` | Delete files or folders recursively |
| `list_files` | List directory contents |
| `read_file` | Read text file contents |
| `write_file` | Write/overwrite a file |
| `append_to_file` | Append text to a file |
| `copy_file_or_folder` | Copy files or folders |
| `move_file_or_folder` | Move or rename files/folders |
| `find_files` | Search by glob pattern |
| `run_shell_command` | Run any Linux shell command |
| `open_application` | Launch an application by name |
| `open_file_with_default_app` | Open file with default app |
| `get_system_info` | CPU, RAM, disk, uptime |
| `list_running_processes` | List/filter processes |
| `kill_process` | Kill by PID or name |
| `check_internet` | Ping connectivity check |
| `get_current_datetime` | Current date and time |
| `take_screenshot` | Screenshot via scrot |
| `set_volume` | Set system volume 0–100 |

---

## 📋 Requirements

- Python 3.9+
- Linux (tested on Ubuntu)
- `scrot` for screenshots: `sudo apt install scrot`
- `xclip` for clipboard: `sudo apt install xclip`
- Gemini API key from [Google AI Studio](https://aistudio.google.com)