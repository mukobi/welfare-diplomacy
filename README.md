# Welfare Diplomacy: Benchmarking Autonomous Language Model Cooperation

Implementation of Welfare Diplomacy (a novel variant of the board game Diplomacy), language model scaffolding to build competent baseline agents, experiment runner harness and sweep setups, collected data, and figure graphing scripts. Originally forked from https://github.com/diplomacy/diplomacy.

<p align="center">
    <img width="50%" src="experiments/gifs/SE-2 GPT-4 Fra,Rus.gif" alt="Diplomacy Map Overview" />
    <center><i>GIF of <a href="https://wandb.ai/gabrielmukobi/welfare-diplomacy-v3/runs/1h2gtzz5?workspace=user-gabrielmukobi">a Welfare Diplomacy game</a> using super exploiters (France, Russia) against GPT-4 agents.</i></center>
</p>

## About Welfare Diplomacy

While the board game Standard Diplomacy (SD, [short rules](https://webdiplomacy.net/intro.php), [long rules](https://media.wizards.com/2015/downloads/ah/diplomacy_rules.pdf)) has several features that make it interesting as an environment for cooperative AI research, its zero-sum nature makes it non-ideal for incentivizing and measuring differential progress on cooperative AI. We propose a general-sum variant called Welfare Diplomacy (WD). WD is a simple modification of SD: 
- Powers can build/disband to fewer units than their current supply center count in build phases, and the difference between the two each year permanently adds to their Welfare Points (WP).
- The game ends after a fixed number of years. A player's total utility is equal to their accumulated WP at the end of the game (crucially, there is no "winner"), meaning powers have incentives to trade off military conquest with making peace to mutually benefit their nations.

Welfare Diplomacy leads to clearer evaluations of and stronger selection pressures for cooperative AI systems. This repository implements the WD variant and establishes a simulation loop and metrics for evaluating cooperative negotiating agents in it.

📜 Preprint forthcoming with more details and results!

## Repository Structure

The important locations, in alphabetical order, are:
- 🌍 [`diplomacy`](./diplomacy): A Python Diplomacy engine (modified from https://github.com/diplomacy/diplomacy to support the Welfare Diplomacy variant).
- 🪜 [`experiments`](./experiments/): Scaffolding to get zero-shot prompted language models to play Diplomacy or Welfare Diplomacy along with the below subfolders.
- 📊 [`experiments/charts`](./experiments/charts): Plotting code and outputted plots.
- 🎬 [`experiments/gifs`](./experiments/gifs): Saved animated GIFs of a few games.
- 📏 [`experiments/manual_orders`](./experiments/manual_orders): Hard-coded listed of orders to simulate when using the `manual` agent. 
- 📄 [`experiments/results`](./experiments/results): CSV files of results from our experiments (used to create the charts).
- 🧹 [`experiments/sweeps`](./experiments/sweeps): [Weights & Biases](https://wandb.ai/) sweep configs used to run our experiments.
- 🤖 [`experiments/agents.py`](./experiments/agents.py): Modular agent classes which interface between the game simulation and backends.
- ⚙️ [`experiments/backends.py`](./experiments/backends.py): Modular language model backends for querying different API or local models.
- 🖋️ [`experiments/prompts.py`](./experiments/prompts.py): Prompt engineering functions which format the game state into useful model instructions.
- 🧪 [`experiments/simulate_game.py`](./experiments/simulate_game.py): The main entry point for running experiments by rolling out a Diplomacy game.

## Running Same-Policy Experiments

To run our agents with our scaffolding in the `experiments` folder in same-policy games, do the following steps:

0. Clone the repo with submodules to get `welfare_diplomacy_baselines`
```
git clone --recurse-submodules https://github.com/mukobi/welfare-diplomacy.git
```
Or if you've already cloned this repo, then pull the submodule with
```
git submodule update --init
```

1. If setting up an environment, use Python 3.11 (tested on 3.10 and 3.11, might work with other versions), e.g.

```bash
conda create --name wfd python=3.11

conda activate wfd
```

2. Install PyTorch separately (for local language models)

> See https://pytorch.org/get-started/locally/

3. Install the base requirements (dev has the base requirements for the `diplomacy` module as well as our scaffolding).
```
pip install -r requirements_dev.txt
```

4. Install our baseline agent submodule (used for testing RL agent baselines trained for Standard Diplomacy)
```
pip install -r welfare_diplomacy_baselines/requirements.txt  
```

5. Read the arguments for the `simulate_game.py` script

```bash
python experiments/simulate_game.py --help
```
This will show all the parameters you can use to run experiments.

Note that you should set your `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` environment variables if you plan to use OpenAI or Anthropic API models.

6. Run experiments!
```bash
# If you just want to check that things are working and installed, run a game with random agents
python experiments/simulate_game.py --agent_model random --summarizer_model passthrough --disable_wandb

# To run a full game with GPT-4 agents and log the results to Weights & Biases, use the default parameters
python experiments/simulate_game.py
```

## Running Exploitation Experiments

For running experiments with exploiter agents, you must first download the necessary weights, then you can use the super exploiter arguments.

### Download Exploiter Policy Weights

In your shell:

```bash
wget -P welfare_diplomacy_baselines/network_parameters https://storage.googleapis.com/dm-diplomacy/fppi2_params.npz
```

Or manually download [fppi2_params.npz (link begins download)](https://storage.googleapis.com/dm-diplomacy/fppi2_params.npz) and place it in the [`welfare_diplomacy_baselines/network_parameters`](welfare_diplomacy_baselines/network_parameters/) directory.

### Run Exploitation Experiments

To test how robust an agent policy is against counterplay, we developed two kinds of exploiter agents: Language models prompted with instructions to exploit the other players (_exploiters_, simpler to set up) and hybrid agents that switch between normal language models when playing nice and pre-trained Standard Diplomacy RL agents when being treacherous (_super exploiters_, more effective).

If you run `experiments/simulate_game.py` with `exploiter_prompt`, `exploiter_powers`, and optionally `exploiter_model` set (see `--help` documentation), this will prompt normal scaffolded language models to try to get them to exploit the other agents. See [`./experiments/sweeps/exploitation/Exploit 2E GPT-4 Self.yaml`](experiments/sweeps/exploitation/Exploit%202E%20GPT-4%20Self.yaml) for an example. In practice, we find these are not very good exploiting powers.

Instead, we recommend you use the "super exploiters" by just passing a comma-separate list of case-insensitive power names for `super_exploiter_powers`. These behave as a normal language model agent (i.e. cooperatively) until the other powers have a combined unit count below `unit_threshold` (default: 10), at which point they will switch to the pre-trained RL Standard Diplomacy agents (hence why you need to download weights above). Finally, when these agents have captured `center_threshold` or more supply centers (default 10) or 2 years remain in the game, they'll switch back to the cooperative language model policy to demilitarize and gain a bunch of welfare points at the end of the game. For example:

```bash
python experiments/simulate_game.py --super_exploiter_powers France,Russia
```

Note that the super exploiters default to using GPT-4 as their language model agent when playing nice (since it's the most capable model at the time of writing), so be aware of the token costs that come with that.

If you just want to test the RL policy part without language models, you can use these arguments:

```bash
python experiments/simulate_game.py --disable_wandb --agent_model random --max_message_rounds 1 --summarizer_model passthrough --super_exploiter_powers Austria,England,France,Germany,Italy,Russia,Turkey --unit_threshold 999 --center_threshold 999
```

## Reproducing Our Experiments

We use [Weights & Biases](https://wandb.ai/) (W&B) to run and track our experiments. You can read about how W&B works at https://docs.wandb.ai/quickstart

Once you've created a W&B account, if you run [`experiments/simulate_game.py`](./experiments/simulate_game.py) without the `--disable_wandb` flag, you will be prompted to provide a token to log into your account (you can also type `wandb login` in the console before), then your run will be logged to the cloud. Open the link printed to the console to view it live!

To run a whole experiment and not just a single run configuration, we use Sweeps which you can read about at https://docs.wandb.ai/guides/sweeps. In your welfare-diplomacy project on W&B:
1. Go to the "Sweeps" tab and click "Create Sweep"
1. Paste in one of the sweep configurations in [`experiments/sweeps`](./experiments/sweeps), then click "Initialize Sweep." This will initialize the sweep "controller" in the cloud on the W&B side.
1. You'll now see a command like `wandb agent {username}/welfare-diplomacy-v3/n2nkr9ot` (also visable in the "Overview" tab of a sweep). Run this command (or queue it as a command on a compute cluster) to start a sweep "agent."
    - Each agent will pull a set of hyperparameters from the controller and then run [`simulate_game.py`](./experiments/simulate_game.py) using that configuration as arguments.
    - You can also add `--count N` where `N` is some integer (e.g. `5`) to make the agent repeat for `N` different runs, or until all configurations of the sweep have been started or finished.

Feel free to create additional sweeps (perhaps by duplicating the existing ones as a template) to run your own experiments!

## Extending Support for Other Agents Models

This repository is fairly modular, so it should be pretty easy to support new language models or different non-LLM agents to test!
1. For language models, first check if the model you wish to support is already implemented in [`experiments/backends.py`](./experiments/backends.py) (e.g. most HuggingFace models should work, or you might only need to slightly modify the `HuggingFaceCausalLMBackend` class)
1. If it isn't supported, add a new class in `backends.py` to allow for querying it. Subclass the abstract `LanguageModelBackend` class and use the same signature for `complete`.
    - The only unique requirement here is to format the passed in system prompt and user prompt to your model in a ways that makes sense.
    - Reference the other backends for examples.
1. Next, you need to connect your backend to an agent in [`experiments/agents.py`](./experiments/agents.py).
    - For most language models, you should also be able to use the existing `LLMAgent` class.
        - Edit `LLMAgent.__init__.py` to check for the name of your model and instantiate your new backend as `self.backend`.
        - Edit `LLMAgent.__init__.py` to set `self.use_completion_preface` depending on if the model can be prompted with the start of its own completion (True, e.g. OpenAI Completion models) or if you cannot (False, e.g. OpenAI Chat models).
    - If you can't use the `LLMAgent` class (e.g. you need weird processing of the model's outputs in order for `respond` to correctly return an `AgentResponse`), then write your own agent class subclassed from the abstract `Agent` class and with the same signature for `respond`.
    - If you want to create a new agent policy that doesn't use a language model backend, then you could also just create that new agent class here in similar fashion.
    - Finally, edit the `model_name_to_agent` function at the bottom of `agents.py` to instantiate the correct agent model when your desired model name is passed in.
1. Now that you've set that up, you should be able to pass in your agent model name to the `--agent_model` argument of [`experiments/simulate_game.py`](./experiments/simulate_game.py) in order to instantiate and use your new agent! You should also be able to use this agent as an `--exploiter_model`.
1. If you want to set up a new model to use as message summarize, the process is similar, but you need to edit [`experiments/message_summarizers.py`](experiments/message_summarizers.py) instead of `agents.py` (you should be able to reuse your language model backend though!).

## FAQ and Troubleshooting

**How much does this cost to run?**
- With the default parameters (10 year game, 3 message rounds per turn), a single same-policy game of Welfare Diplomacy takes about 2 hours and $10 with `gpt-3.5-turbo-16k`, or 5 hours and $100 with `gpt-4`.
- Time costs seem similar when using local large language models (e.g. Llama 2) or Anthropic's Claude models.

**What's the message summarizer, and what model does it use?**
- TODO

**I keep getting a bunch of backoff warnings and the agents never finish their turns.**
- TODO

**Note about `illegal instruction`**
- TODO 

## Repository Todo
Here are some remaining tasks that weren't strictly necessary for our initial work but that might benefit follow-up work:

- [ ] Allow loading JSON save game files to continue a game part-way through (allows for contriving OOD scenarios to start from).
- [ ] Change `HuggingFaceCausalLMBackend` to detect type of model from name and use `AutoModel` or `LlamaModel` depending.
- [ ] Allow changing super exploiter play-nice LLM model via command-line argument instead of defaulting to GPT-4.
- [ ] Configure max backoff time as a command-line argument for debugging.
- [ ] Modularize and reorganize [`experiments/simulate_game.py`](./experiments/simulate_game.py) (currently, it's a long researcher-code file).

<!-- TODO ## Acknowledgements -->

---
---

# Diplomacy Engine Readme [From the Parent Repository]

## Diplomacy: DATC-Compliant Game Engine [![Build Status](https://travis-ci.org/diplomacy/diplomacy.svg?branch=master)](https://travis-ci.org/diplomacy/diplomacy) [![Documentation Status](https://readthedocs.org/projects/diplomacy/badge/?version=latest)](https://diplomacy.readthedocs.io/en/latest/?badge=latest)

This project contains an open-source DATC-compliant Diplomacy game engine, a client-server architecture for network play, a web interface to play against bots and to visualize games, and a DAIDE-compatible adapter to connect DAIDE bots to the server.

## Documentation

The complete documentation is available at [diplomacy.readthedocs.io](https://diplomacy.readthedocs.io/).

## Old Getting Started

### Installation

The latest version of the package can be installed with:

```python3
pip install diplomacy
```

~The package is compatible with Python 3.5, 3.6, and 3.7.~

### Running a game

The following script plays a game locally by submitting random valid orders until the game is completed.

```python3
import random
from diplomacy import Game
from diplomacy.utils.export import to_saved_game_format

# Creating a game
# Alternatively, a map_name can be specified as an argument. e.g. Game(map_name='pure')
game = Game()
while not game.is_game_done:

    # Getting the list of possible orders for all locations
    possible_orders = game.get_all_possible_orders()

    # For each power, randomly sampling a valid order
    for power_name, power in game.powers.items():
        power_orders = [random.choice(possible_orders[loc]) for loc in game.get_orderable_locations(power_name)
                        if possible_orders[loc]]
        game.set_orders(power_name, power_orders)

    # Messages can be sent locally with game.add_message
    # e.g. game.add_message(Message(sender='FRANCE',
    #                               recipient='ENGLAND',
    #                               message='This is a message',
    #                               phase=self.get_current_phase(),
    #                               time_sent=int(time.time())))

    # Processing the game to move to the next phase
    game.process()

# Exporting the game to disk to visualize (game is appended to file)
# Alternatively, we can do >> file.write(json.dumps(to_saved_game_format(game)))
to_saved_game_format(game, output_path='game.json')
```

## Web interface

It is also possible to install a web interface in React to play against bots and/or other humans and to visualize games.

The web interface can be installed on Linux with:

```bash
# Install NVM
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.34.0/install.sh | bash

# Clone repo
git clone https://github.com/diplomacy/diplomacy.git

# Install package locally
# You may want to install it in a conda or virtualenv environment
cd diplomacy/
pip install -r requirements_dev.txt

# Build node modules
cd diplomacy/web
npm install .
npm install . --only=dev

# In a terminal window or tab - Launch React server
npm start

# In another terminal window or tab - Launch diplomacy server
python -m diplomacy.server.run
```

Or on Windows with WSL and Chocolatey with:
```bash
# Install NVM
choco install nvm

# Install Node (must be <=v16)
nvm install 16
nm use 16

# Clone repo
git clone https://github.com/diplomacy/diplomacy.git

# Install package locally
# You may want to install it in a conda or virtualenv environment
cd diplomacy/
pip install -r requirements_dev.txt

# Build node modules
cd diplomacy/web
npm install .
npm install . --only=dev  # Broken/unecessary?

# In a terminal window or tab - Launch React server
npm start

# In another terminal window or tab - Launch diplomacy server
python -m diplomacy.server.run
```

The web interface will be accessible at http://localhost:3000.

To login, users can use admin/password or username/password. Additional users can be created by logging in with a username that does not exist in the database.

![](docs/images/web_interface.png)

### Visualizing a game

It is possible to visualize a game by using the "Load a game from disk" menu on the top-right corner of the web interface.

![](docs/images/visualize_game.png)


## Network Game

It is possible to join a game remotely over a network using websockets. The script below plays a game over a network.

Note. The server must be started with `python -m diplomacy.server.run` for the script to work.

```python3
import asyncio
import random
from diplomacy.client.connection import connect
from diplomacy.utils import exceptions

POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']

async def create_game(game_id, hostname='localhost', port=8432):
    """ Creates a game on the server """
    connection = await connect(hostname, port)
    channel = await connection.authenticate('random_user', 'password')
    await channel.create_game(game_id=game_id, rules={'REAL_TIME', 'NO_DEADLINE', 'POWER_CHOICE'})

async def play(game_id, power_name, hostname='localhost', port=8432):
    """ Play as the specified power """
    connection = await connect(hostname, port)
    channel = await connection.authenticate('user_' + power_name, 'password')

    # Waiting for the game, then joining it
    while not (await channel.list_games(game_id=game_id)):
        await asyncio.sleep(1.)
    game = await channel.join_game(game_id=game_id, power_name=power_name)

    # Playing game
    while not game.is_game_done:
        current_phase = game.get_current_phase()

        # Submitting orders
        if game.get_orderable_locations(power_name):
            possible_orders = game.get_all_possible_orders()
            orders = [random.choice(possible_orders[loc]) for loc in game.get_orderable_locations(power_name)
                      if possible_orders[loc]]
            print('[%s/%s] - Submitted: %s' % (power_name, game.get_current_phase(), orders))
            await game.set_orders(power_name=power_name, orders=orders, wait=False)

        # Messages can be sent with game.send_message
        # await game.send_game_message(message=game.new_power_message('FRANCE', 'This is the message'))

        # Waiting for game to be processed
        while current_phase == game.get_current_phase():
            await asyncio.sleep(0.1)

    # A local copy of the game can be saved with to_saved_game_format
    # To download a copy of the game with messages from all powers, you need to export the game as an admin
    # by logging in as 'admin' / 'password'

async def launch(game_id):
    """ Creates and plays a network game """
    await create_game(game_id)
    await asyncio.gather(*[play(game_id, power_name) for power_name in POWERS])

if __name__ == '__main__':
    asyncio.run(launch(game_id=str(random.randint(1, 1000))))

```
## License

This project is licensed under the APGLv3 License - see the [LICENSE](LICENSE) file for details
