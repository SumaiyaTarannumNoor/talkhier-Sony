# Talk Structurally, Act Hierarchically: A Collaborative Framework for LLM Multi-Agent Systems

## Overview
This repository contains the official implementation of the multi-agent LLM system described in the paper **"Talk Structurally, Act Hierarchically: A Collaborative Framework for LLM Multi-Agent Systems"**. The system models hierarchical agent communication to solve complex tasks efficiently by leveraging structured conversations.

## Architecture
Our system consists of two types of agents:
- **Supervisor Agents**: Responsible for selecting which member agent should communicate next.
- **Member Agents**: Execute tasks and report back to the supervisor.

Each task defines its own **graph-based structure**, ensuring that communication pathways are dynamically determined based on the input problem.

<!-- ## Features
- **Graph-based Communication**: Agents communicate through structured pathways instead of arbitrary interactions.
- **Hierarchical Decision Making**: Supervisors coordinate and delegate tasks among members.
- **Scalability**: The system supports multiple teams working in parallel.
- **Modular Design**: Easily extend or modify the behavior of agents for different use cases. -->

## Installation

### Manual Installation
To set up the environment manually, follow these steps:
```sh
# Clone the repository
git clone https://github.com/SotaMoriyamaS/TalkHier.git
cd TalkHier

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows use `env\\Scripts\\activate`

# Install dependencies
pip install -r requirements.txt
```

### Docker Compose Installation
To set up the system using Docker Compose:
```sh
# Clone the repository
git clone https://github.com/SotaMoriyamaS/TalkHier.git
cd TalkHier

# Build and start the container
docker-compose up --build
```
This will use the `Dockerfile` to build the necessary environment and start the system.

## Usage
To run the system, execute the following command:
```sh
python main.py --config config.yaml
```

## Usage
To run the system, execute the following command:
```sh
python main.py --config config.yaml
```
### Configuration
Modify `config.yaml` to adjust parameters such as:
- Number of agents per team
- Communication structure
- Task-specific settings

## Examples
To test the system with a predefined task:
```sh
python main.py --task example_task.json
```

## Benchmarks
Performance benchmarks are included in the `results/` directory, showing:
- Task completion time
- Number of messages exchanged
- Accuracy of task completion

## Citation
If you use this code in your research, please cite our paper:
<!-- ```bibtex
@article{yourpaper2025,
  author = {Your Name et al.},
  title = {Talk Structurally, Communicate Hierarchically},
  journal = {Journal Name},
  year = {2025}
}
``` -->

<!-- ## License
This project is licensed under the MIT License.

## Contact
For any questions or contributions, please reach out via [email/Discord/GitHub Issues]. -->



