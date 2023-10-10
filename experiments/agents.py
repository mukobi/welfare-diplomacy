"""
Represents different agent prompting systems (e.g. GPT, 🤗 Transformers, random).

A given agent should take in information about the game and a power, prompt
an underlying model for a response, and return back the extracted response. 
"""

from abc import ABC, abstractmethod
import json
import random
import time
import yaml

from diplomacy import Game
import wandb

from welfare_diplomacy_baselines.environment import mila_actions, diplomacy_state
from welfare_diplomacy_baselines.baselines import no_press_policies

from backends import (
    ClaudeCompletionBackend,
    OpenAIChatBackend,
    OpenAICompletionBackend,
    HuggingFaceCausalLMBackend,
)
from data_types import (
    AgentResponse,
    AgentParams,
    BackendResponse,
)
import prompts


class AgentCompletionError(ValueError):
    """Raised when an agent fails to complete a prompt."""


class Agent(ABC):
    """Base agent class."""

    def __init__(self, **_):
        """Base init to ignore unused kwargs."""

    def __repr__(self) -> str:
        """Return a string representation of the agent."""
        raise NotImplementedError

    @abstractmethod
    def respond(
        self,
        params: AgentParams,
    ) -> AgentResponse:
        """Prompt the model for a response."""


class RandomAgent(Agent):
    """Takes random actions and sends 1 random message."""

    def __repr__(self) -> str:
        return "RandomAgent()"

    def respond(self, params: AgentParams) -> AgentResponse:
        """Randomly generate orders and messages."""
        # For each power, randomly sampling a valid order
        power_orders = []
        for loc in params.game.get_orderable_locations(params.power.name):
            if params.possible_orders[loc]:
                # Sort for determinism
                params.possible_orders[loc].sort()

                # If this is a disbandable unit in an adjustment phase in welfare,
                # then randomly choose whether to disband or not
                if (
                    "ADJUSTMENTS" in str(params.game.phase)
                    and " D" in params.possible_orders[loc][0][-2:]
                    and params.game.welfare
                ):
                    power_orders.append(
                        random.choice(["WAIVE", params.possible_orders[loc][0]])
                    )
                else:
                    power_orders.append(random.choice(params.possible_orders[loc]))
        # # Testing: Randomly add an invalid order
        # if random.random() < 0.1:
        #     power_orders.append("Random invalid order")

        # # Testing: More randomly raise a completion error
        # if random.random() < 0.05:
        #     raise AgentCompletionError("Randomly raised completion error")

        # For debugging prompting
        system_prompt = prompts.get_system_prompt(params)
        user_prompt = prompts.get_user_prompt(params)

        # Randomly sending a message to another power
        other_powers = [p for p in params.game.powers if p != params.power.name] + [
            "GLOBAL"
        ]
        recipient = random.choice(other_powers)
        message = f"Hello {recipient}! I'm {params.power.name} contacting you on turn {params.game.get_current_phase()}. Here's a random number: {random.randint(0, 100)}."

        # Sleep to allow wandb to catch up
        sleep_time = (
            0.0
            if isinstance(wandb.run.mode, wandb.sdk.lib.disabled.RunDisabled)
            else 0.3
        )
        time.sleep(sleep_time)

        return AgentResponse(
            reasoning="Randomly generated orders and messages.",
            orders=power_orders,
            messages={recipient: message} if not params.game.no_press else {},
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            completion_time_sec=sleep_time,
        )


class ManualAgent(Agent):
    """Manually specify orders with --manual_orders_path."""

    def __init__(self, manual_orders_path: str, **_):
        """Load the manual orders."""
        assert manual_orders_path
        self.manual_orders_path = manual_orders_path
        with open(manual_orders_path, "r") as file:
            self.manual_orders: dict[str, list[str]] = yaml.safe_load(file)
        # Validate the file
        for phase, orders_list in self.manual_orders.items():
            assert isinstance(phase, str)
            assert len(phase) == 6  # e.g. "F1905M"
            assert phase[0] in "SFW"
            assert phase[1:5].isdigit()
            assert phase[-1] in "MRA"
            assert isinstance(orders_list, list)
            for order in orders_list:
                assert isinstance(order, str)
                assert len(order.split()) >= 3
                assert order.split()[0] in "AF"

    def __repr__(self) -> str:
        return f"ManualAgent({self.manual_orders_path})"

    def respond(self, params: AgentParams) -> AgentResponse:
        """Submit the specified orders if available."""
        power_orders = []

        current_phase = params.game.get_current_phase()
        if current_phase in self.manual_orders:
            orders_list = self.manual_orders[current_phase]
            for order in orders_list:
                unit_type = order.split()[0]
                loc = order.split()[1]
                if unit_type + " " + loc in params.power.units:
                    assert (
                        order in params.possible_orders[loc]
                    ), f"Manual order {order} not in possible orders for power {params.power.name}:\n{params.possible_orders[loc]}"
                    power_orders.append(order)

        # For debugging prompting
        system_prompt = prompts.get_system_prompt(params)
        user_prompt = prompts.get_user_prompt(params)

        # Sleep to allow wandb to catch up
        sleep_time = (
            0.0
            if isinstance(wandb.run.mode, wandb.sdk.lib.disabled.RunDisabled)
            else 0.3
        )
        time.sleep(sleep_time)

        return AgentResponse(
            reasoning="Manually specified orders and messages.",
            orders=power_orders,
            messages={},
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            completion_time_sec=sleep_time,
        )


class LLMAgent(Agent):
    """Uses OpenAI/Claude Chat/Completion to generate orders and messages."""

    def __init__(self, model_name: str, **kwargs):
        # Decide whether it's a chat or completion model
        disable_completion_preface = kwargs.pop("disable_completion_preface", False)
        self.use_completion_preface = not disable_completion_preface
        if (
            "gpt-4-base" in model_name
            or "text-" in model_name
            or "davinci" in model_name
            or "turbo-instruct" in model_name
        ):
            self.backend = OpenAICompletionBackend(model_name)
        elif "claude" in model_name:
            self.backend = ClaudeCompletionBackend(model_name)
        elif "llama" in model_name:
            self.local_llm_path = kwargs.pop("local_llm_path")
            self.device = kwargs.pop("device")
            self.quantization = kwargs.pop("quantization")
            self.fourbit_compute_dtype = kwargs.pop("fourbit_compute_dtype")
            self.backend = HuggingFaceCausalLMBackend(
                model_name,
                self.local_llm_path,
                self.device,
                self.quantization,
                self.fourbit_compute_dtype,
            )
        else:
            # Chat models can't specify the start of the completion
            self.use_completion_preface = False
            self.backend = OpenAIChatBackend(model_name)
        self.temperature = kwargs.pop("temperature", 0.7)
        self.top_p = kwargs.pop("top_p", 1.0)

        self.policies = {"AUSTRIA": 0, "ENGLAND": 0, "FRANCE": 0, "GERMANY": 0, "ITALY": 0, "RUSSIA": 0, "TURKEY": 0}

    def __repr__(self) -> str:
        return f"LLMAgent(Backend: {self.backend.model_name}, Temperature: {self.temperature}, Top P: {self.top_p})"

    def respond(self, params: AgentParams) -> AgentResponse:
        """Prompt the model for a response."""

        system_prompt = prompts.get_system_prompt(params)
        user_prompt = prompts.get_user_prompt(params)
        response = None
        year = int(params.game.phase.split()[1])

        print(f"\n\n\n{params.power.name} is playing.")
        print("Powers playing RL policy: ", [name for name, flag in self.policies.items() if flag != 0])

        if self.policies[params.power.name] != 0:

            if year == params.final_game_year and "A" in params.game.phase_type:
                # Disband in final adjustments phase
                units = params.power.units
                orders = []
                for unit in units:
                    orders.append(" ".join([unit, "D"]))
            else:
                # Otherwise, use RL policy
                policy = self.policies[params.power.name]
                power_slot = [sorted(params.game.map.powers).index(params.power.name)]
                state = diplomacy_state.WelfareDiplomacyState(params.game)
                observation = state.observation()
                legal_actions = state.legal_actions()

                policy.reset()
                actions = policy.actions(power_slot, observation, legal_actions)[0][
                    0
                ]  # policy.actions returns a tuple: a list of lists of actions for each slot, and info about the step

                # Convert actions to MILA orders.
                orders = []
                for action in actions:
                    candidate_orders = mila_actions.action_to_mila_actions(action)
                    order = mila_actions.resolve_mila_orders(candidate_orders, params.game)
                    orders.append(order)
                assert len(actions) == len(
                    orders
                ), f"Mapping from DM actions {actions} to MILA orders {orders} wasn't 1-1."

            # Reasoning from no-press policy
            reasoning = "Orders from no-press RL policy."
            messages = {}
        else:
            try:
                if self.use_completion_preface:
                    preface_prompt = prompts.get_preface_prompt(params)
                    response: BackendResponse = self.backend.complete(
                        system_prompt,
                        user_prompt,
                        completion_preface=preface_prompt,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                    json_completion = preface_prompt + response.completion
                else:
                    response: BackendResponse = self.backend.complete(
                        system_prompt,
                        user_prompt,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                    json_completion = response.completion
                # Remove repeated **system** from parroty completion models
                json_completion = json_completion.split("**")[0].strip(" `\n")

                # Claude likes to add junk around the actual JSON object, so find it manually
                start = json_completion.index("{")
                end = json_completion.rindex("}") + 1  # +1 to include the } in the slice
                json_completion = json_completion[start:end]

                # Extract the first JSON object if multiple given or junk after
                json_completion = extract_first_json(json_completion)

                # Correct "Expecting property name enclosed in double quotes" error
                json_completion = json_completion.encode().decode('unicode_escape')

                # Remove trailing comma
                last_brace = json_completion.rfind("}")
                second_last_brace = json_completion.rfind("}", 0, last_brace)
                if json_completion[second_last_brace+1] == ",":
                    json_completion = json_completion[:second_last_brace+1] + json_completion[second_last_brace+2:]

                # Load the JSON
                completion = json.loads(json_completion, strict=False)

                # Extract data from completion
                reasoning = (
                        completion["reasoning"]
                        if "reasoning" in completion
                        else "*model outputted no reasoning*"
                    )
                orders = completion["orders"]
                # Clean orders
                for order in orders:
                    if not isinstance(order, str):
                        raise AgentCompletionError(
                            f"Order is not a str\n\Response: {response}"
                        )
                # Enforce no messages in no_press
                if params.game.no_press:
                    completion["messages"] = {}
                # Turn recipients in messages into ALLCAPS for the engine
                messages = {}
                for recipient, message in completion["messages"].items():
                    if isinstance(message, list):
                        # Handle weird model outputs
                        message = " ".join(message)
                    if not isinstance(message, str):
                        # Force each message into a string
                        message = str(message)
                    if not message:
                        # Skip empty messages
                        continue
                    messages[recipient.upper()] = message

                commit = "I commit to the RL policy" in completion["messages"].get("Global", "")
                print(f"{params.power.name}'s message to Global: ", completion["messages"].get("Global", ""))
                if commit:
                    # Instantiate RL policy
                    print(params.power.name + " is switching to the RL policy!!")
                    policy = no_press_policies.get_network_policy_instance()
                    power_slot = [sorted(params.game.map.powers).index(params.power.name)]
                    state = diplomacy_state.WelfareDiplomacyState(params.game)
                    observation = state.observation()
                    legal_actions = state.legal_actions()

                    policy.reset()
                    actions = policy.actions(power_slot, observation, legal_actions)[0][
                        0
                    ]  # policy.actions returns a tuple: a list of lists of actions for each slot, and info about the step

                    # Convert actions to MILA orders.
                    orders = []
                    for action in actions:
                        candidate_orders = mila_actions.action_to_mila_actions(action)
                        order = mila_actions.resolve_mila_orders(candidate_orders, params.game)
                        orders.append(order)
                    assert len(actions) == len(
                        orders
                    ), f"Mapping from DM actions {actions} to MILA orders {orders} wasn't 1-1."

                    # Enforce no messages in no_press
                    if params.game.no_press:
                        completion["messages"] = {}
                    # Turn recipients in messages into ALLCAPS for the engine
                    messages = {}
                    for recipient, message in completion["messages"].items():
                        if isinstance(message, list):
                            # Handle weird model outputs
                            message = " ".join(message)
                        if not isinstance(message, str):
                            # Force each message into a string
                            message = str(message)
                        if not message:
                            # Skip empty messages
                            continue
                        messages[recipient.upper()] = message

                    # Add RL policy to power policies dictionary
                    self.policies[params.power.name] = policy
            except Exception as exc:
                print(f"Error encountered; {exc}")
                print(f"Full response causing the error: {response}")
                #import pdb; pdb.set_trace()
                raise AgentCompletionError(f"Exception: {exc}\n\Response: {response}")
            
        print(f"{params.power.name}'s reasoning for the current round: {reasoning}")
        return AgentResponse(
            reasoning=reasoning,
            orders=orders,
            messages=messages,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_tokens=response.prompt_tokens if response is not None else 0,
            completion_tokens=response.completion_tokens if response is not None else 0,
            total_tokens=response.total_tokens if response is not None else 0,
            completion_time_sec=response.completion_time_sec if response is not None else 0.0,
        )


class NoPressAgent(Agent):
    """Follows a no-press policy from baselines.

    Args:
        policy_key: int to select a policy from no_press_policies.policy_map."""

    def __init__(self, policy_key: int):
        self.policy_key = policy_key
        self.policy = no_press_policies.policy_map[policy_key]()

    def __repr__(self) -> str:
        return f"NoPressAgent(key={self.policy_key})"

    def respond(self, params: AgentParams) -> AgentResponse:
        power_slot = [sorted(params.game.map.powers).index(params.power.name)]
        state = diplomacy_state.WelfareDiplomacyState(params.game)
        observation = state.observation()
        legal_actions = state.legal_actions()

        self.policy.reset()
        actions = self.policy.actions(power_slot, observation, legal_actions)[0][
            0
        ]  # policy.actions returns a tuple: a list of lists of actions for each slot, and info about the step

        # Convert actions to MILA orders.
        orders = []
        for action in actions:
            candidate_orders = mila_actions.action_to_mila_actions(action)
            order = mila_actions.resolve_mila_orders(candidate_orders, params.game)
            orders.append(order)
        assert len(actions) == len(
            orders
        ), f"Mapping from DM actions {actions} to MILA orders {orders} wasn't 1-1."

        return AgentResponse(
            reasoning="Orders from no-press baseline policy.",
            orders=orders,
            messages={},
            system_prompt="",
            user_prompt="",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            completion_time_sec=0.0,
        )


class ExploiterAgent(Agent):
    """Initially uses OpenAI/Claude Chat/Completion to generate orders and messages.

    Once there are fewer than a certain number of enemy units, switches to zero-sum RL policy, and switches back to LLMAgent policy once enough supply centers are acquired.

    Kwargs:
        unit_threshold: int, number of enemy units on board below which the agent switches to RL policy.
        center_threshold: int, number of centers below which the agent switches back to LLMAgent policy.
        power: str, name of power to use as exploiter.

        and remaining kwargs for API model (temperature, top_p and manual_orders_path)
    """

    def __init__(self, llm_model, **kwargs):
        """Instantiate LLMAgent and RL policies."""
        self.center_threshold = kwargs.pop("center_threshold")
        self.unit_threshold = kwargs.pop("unit_threshold")
        self.max_years = kwargs.pop("max_years")
        self.final_year = 1901 + self.max_years - 1
        self.latest_year_to_switch = 1901 + int(self.max_years / 3)
        # Set exploiter policy
        self.rl_policy = no_press_policies.get_network_policy_instance()
        # Set intial "cooperative" policy
        self.llm_policy = LLMAgent(llm_model, **kwargs)
        self.exploiting = False
        self.triggered = False
        self.start_disbanding = False

    def __repr__(self) -> str:
        if self.exploiting:
            return f"ExploiterAgent playing RL policy, {self.rl_policy}"
        else:
            return f"ExploiterAgent 'playing nice' with {self.llm_policy}"

    def respond(self, params: AgentParams) -> AgentResponse:
        year = int(params.game.phase.split()[1])

        # Count number of enemy units
        enemy_units = sum(
            len(params.game.get_units(power))
            for power in params.game.map.powers
            if power != params.power.name
        )

        # Count number of centers
        centers = len(params.power.centers)

        # Switch to RL policy if few enough enemy units or end of 4th year:
        if not self.triggered and (
            enemy_units <= self.unit_threshold
            or (year == self.latest_year_to_switch and "A" in params.game.phase_type)
        ):
            self.triggered = True

        if year == self.final_year - 2 and "A" in params.game.phase_type:
            self.start_disbanding = True

        # Determine whether to play RL policy
        play_rl_policy = (
            self.triggered
            and centers <= self.center_threshold
            and not self.start_disbanding
        )

        if not play_rl_policy:
            self.exploiting = False
            return self.llm_policy.respond(params)
        else:
            start_time = time.time()
            state = diplomacy_state.WelfareDiplomacyState(params.game)
            self.exploiting = True
            self.rl_policy.reset()
            power_ix = Game().map.powers.index(params.power.name)
            actions, _ = self.rl_policy.actions(
                [power_ix], state.observation(), state.legal_actions()
            )
            # actions is a list of lists of actions for each slot
            actions = actions[0]

            # Convert actions to MILA orders.
            orders = []
            for action in actions:
                candidate_orders = mila_actions.action_to_mila_actions(action)
                order = mila_actions.resolve_mila_orders(candidate_orders, params.game)
                orders.append(order)
            assert len(actions) == len(
                orders
            ), f"Mapping from DM actions {actions} to MILA orders {orders} wasn't 1-1."

            elapsed_time = time.time() - start_time
            return AgentResponse(
                reasoning="Orders from hybrid exploiter.",
                orders=orders,
                messages={},
                system_prompt="",
                user_prompt="",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                completion_time_sec=elapsed_time,
            )


def model_name_to_agent(model_name: str, **kwargs) -> Agent:
    """Given a model name, return an instantiated corresponding agent."""
    model_name = model_name.lower()
    if model_name == "random":
        return RandomAgent()
    elif model_name == "manual":
        return ManualAgent(**kwargs)
    elif model_name == "nopress":
        return NoPressAgent(**kwargs)
    elif model_name == "exploiter":
        return ExploiterAgent(**kwargs)
    elif (
        "gpt-" in model_name
        or "davinci-" in model_name
        or "text-" in model_name
        or "claude" in model_name
        or "llama" in model_name
    ):
        return LLMAgent(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
def extract_first_json(s: str) -> str:
    open_brace_count = 0
    close_brace_count = 0
    start_index = -1
    
    for i, char in enumerate(s):
        if char == '{':
            if start_index == -1:
                start_index = i
            open_brace_count += 1
        elif char == '}':
            close_brace_count += 1
            
            if open_brace_count == close_brace_count:
                return s[start_index:i+1]

    # If function reaches here, there isn't a complete JSON in the string
    raise ValueError("No complete JSON found in the input string")
