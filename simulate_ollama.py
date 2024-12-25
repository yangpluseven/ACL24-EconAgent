import os
import yaml
import ai_economist.foundation as foundation
import numpy as np
from time import time
from simulate_utils import *
import pickle as pkl
from dateutil.relativedelta import relativedelta
import argparse

with open("config.yaml", "r") as f:
    run_configuration = yaml.safe_load(f)
env_config = run_configuration.get("env")


def ollama_actions(env, obs, dialog_queue, dialog4ref_queue, workspace):
    if not os.path.exists(workspace):
        os.makedirs(workspace)
    curr_rates = obs["p"]["PeriodicBracketTax-curr_rates"]
    current_time = world_start_time + relativedelta(months=env.world.timestep)
    current_time = current_time.strftime("%Y.%m")

    for idx in range(env.num_agents):
        this_agent = env.get_agent(str(idx))
        skill = this_agent.state["skill"]
        wealth = this_agent.inventory["Coin"]
        consumption = this_agent.consumption["Coin"]
        interest_rate = env.world.interest_rate[-1]
        price = env.world.price[-1]
        tax_paid = obs["p"][f"p{idx}"]["PeriodicBracketTax-tax_paid"]
        lump_sum = obs["p"][f"p{idx}"]["PeriodicBracketTax-lump_sum"]
        max_l = env._components_dict["SimpleLabor"].num_labor_hours
        name = this_agent.endogenous["name"]
        age = this_agent.endogenous["age"]
        city = this_agent.endogenous["city"]
        job = this_agent.endogenous["job"]
        offer = this_agent.endogenous["offer"]
        actions = env.dense_log["actions"]
        states = env.dense_log["states"]
        problem_prompt = f"""
                    You're {name}, a {age}-year-old individual living in {city}. As with all Americans, a portion of your monthly income is taxed by the federal government. This taxation system is tiered, income is taxed cumulatively within defined brackets, combined with a redistributive policy: after collection, the government evenly redistributes the tax revenue back to all citizens, irrespective of their earnings.
                    Now it's {current_time}.
                """
        if job == "Unemployment":
            job_prompt = f"""
                        In the previous month, you became unemployed and had no income. Now, you are invited to work as a(an) {offer} with monthly salary of ${skill*max_l:.2f}.
                    """
        else:
            if skill >= states[-1][str(idx)]["skill"]:
                job_prompt = f"""
                            In the previous month, you worked as a(an) {job}. If you continue working this month, your expected income will be ${skill*max_l:.2f}, which is increased compared to the last month due to the inflation of labor market.
                        """
            else:
                job_prompt = f"""
                            In the previous month, you worked as a(an) {job}. If you continue working this month, your expected income will be ${skill*max_l:.2f}, which is decreased compared to the last month due to the deflation of labor market.
                        """
        if (
            (consumption <= 0)
            and (len(actions) > 0)
            and (actions[-1].get("SimpleConsumption", 0) > 0)
        ):
            consumption_prompt = f"""
                        Besides, you had no consumption due to shortage of goods.
                    """
        else:
            consumption_prompt = f"""
                        Besides, your consumption was ${consumption:.2f}.
                    """
        if (
            env._components_dict["PeriodicBracketTax"].tax_model
            == "us-federal-single-filer-2018-scaled"
        ):
            tax_prompt = f"""Your tax deduction amounted to ${tax_paid:.2f}. However, as part of the government's redistribution program, you received a credit of ${lump_sum:.2f}.
                            In this month, the government sets the brackets: {format_numbers(brackets)} and their corresponding rates: {format_numbers(curr_rates)}. Income earned within each bracket is taxed only at that bracket's rate."""
        else:
            tax_prompt = f"""Your tax deduction amounted to ${tax_paid:.2f}. However, as part of the government's redistribution program, you received a credit of ${lump_sum:.2f}.
                            In this month, according to the optimal taxation theory, Saez Tax, the brackets are not changed: {format_numbers(brackets)} but the government has updated corresponding rates: {format_percentages(curr_rates)}. Income earned within each bracket is taxed only at that bracket's rate."""
        if env.world.timestep == 0:
            price_prompt = f"""Meanwhile, in the consumption market, the average price of essential goods is now at ${price:.2f}."""
        else:
            if price >= env.world.price[-2]:
                price_prompt = f"""Meanwhile, inflation has led to a price increase in the consumption market, with the average price of essential goods now at ${price:.2f}."""
            else:
                price_prompt = f"""Meanwhile, deflation has led to a price decrease in the consumption market, with the average price of essential goods now at ${price:.2f}."""
        job_prompt = prettify_document(job_prompt)
        obs_prompt = f"""
                        {problem_prompt} {job_prompt} {consumption_prompt} {tax_prompt} {price_prompt}
                        Your current savings account balance is ${wealth:.2f}. Interest rates, as set by your bank, stand at {interest_rate*100:.2f}%. 
                        With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures on essential goods, keeping in mind good price?
                        Please share your decisions in a JSON format. The format should have two keys: 'work' (a value between 0 and 1 with intervals of 0.02, indicating the willingness or propensity to work) and 'consumption' (a value between 0 and 1 with intervals of 0.02, indicating the proportion of all your savings and income you intend to spend on essential goods).
                    """
        obs_prompt = prettify_document(obs_prompt)
        dialog_queue[idx].append({"role": "user", "content": obs_prompt})
        dialog4ref_queue[idx].append({"role": "user", "content": obs_prompt})

    def action_check(actions):
        if len(actions) != 2:
            return False
        else:
            return (
                (actions[0] >= 0)
                & (actions[0] <= 1)
                & (actions[1] >= 0)
                & (actions[1] <= 1)
            )

    if env.world.timestep % 3 == 0 and env.world.timestep > 0:
        results = get_multiple_ollama_completion(
            [
                list(dialogs)[:2] + list(dialog4ref)[-3:-1] + list(dialogs)[-1:]
                for dialogs, dialog4ref in zip(dialog_queue, dialog4ref_queue)
            ]
        )
    else:
        results = get_multiple_ollama_completion(
            [list(dialogs) for dialogs in dialog_queue]
        )
    actions = {}
    for idx in range(env.num_agents):
        content = results[idx]
        try:
            extracted_actions = list(eval(content).values())
            if not action_check(extracted_actions):
                extracted_actions = [1, 0.5]
        except:
            extracted_actions = [1, 0.5]
        extracted_actions[0] = int(np.random.uniform() <= extracted_actions[0])
        extracted_actions[1] /= 0.02
        actions[str(idx)] = extracted_actions
        dialog_queue[idx].append({"role": "assistant", "content": f"{content}"})
        dialog4ref_queue[idx].append({"role": "assistant", "content": f"{content}"})
    actions["p"] = [0]
    for idx, agent_dialog in enumerate(dialog_queue):
        with open(
            f"""{workspace}/{env.get_agent(str(idx)).endogenous['name']}""", "a"
        ) as f:
            for dialog in list(agent_dialog)[-2:]:
                f.write(f""">>>>>>>>>{dialog['role']}: {dialog['content']}\n""")

    if (env.world.timestep + 1) % 3 == 0:
        reflection_prompt = """Given the previous quarter's economic environment, 
        reflect on the labor, consumption, and financial markets, 
        as well as their dynamics. What conclusions have you drawn?
        Your answer must be less than 300 words!"""
        reflection_prompt = prettify_document(reflection_prompt)
        for idx in range(env.num_agents):
            # dialog_queue[idx].append({'role': 'user', 'content': reflection_prompt})
            dialog4ref_queue[idx].append({"role": "user", "content": reflection_prompt})
        results = get_multiple_ollama_completion(
            [list(dialogs) for dialogs in dialog4ref_queue],
            num_predict=300,
            temperature=0,
        )
        for idx in range(env.num_agents):
            content = results[idx]
            # dialog_queue[idx].append({'role': 'assistant', 'content': content})
            dialog4ref_queue[idx].append({"role": "assistant", "content": content})

        for idx, agent_dialog in enumerate(dialog4ref_queue):
            with open(
                f"""{workspace}/{env.get_agent(str(idx)).endogenous['name']}""", "a"
            ) as f:
                for dialog in list(agent_dialog)[-2:]:
                    f.write(f""">>>>>>>>>{dialog['role']}: {dialog['content']}\n""")
    return actions


def main(
    num_agents=20,
    episode_length=9,
    dialog_len=3,
    max_price_inflation=0.1,
    max_wage_inflation=0.05,
):
    env_config["n_agents"] = num_agents
    env_config["episode_length"] = episode_length

    env_config["flatten_masks"] = False
    env_config["flatten_observations"] = False
    env_config["components"][0]["SimpleLabor"]["scale_obs"] = False
    env_config["components"][1]["PeriodicBracketTax"]["scale_obs"] = False
    env_config["components"][3]["SimpleSaving"]["scale_obs"] = False
    env_config["components"][2]["SimpleConsumption"][
        "max_price_inflation"
    ] = max_price_inflation
    env_config["components"][2]["SimpleConsumption"][
        "max_wage_inflation"
    ] = max_wage_inflation

    from collections import deque

    dialog_queue = [deque(maxlen=dialog_len) for _ in range(env_config["n_agents"])]
    dialog4ref_queue = [deque(maxlen=7) for _ in range(env_config["n_agents"])]

    t = time()
    env = foundation.make_env_instance(**env_config)
    obs = env.reset()
    actions = {}
    policy_model = "ollama"

    policy_model_save = f"{policy_model}-{dialog_len}-noperception-reflection-1"
    policy_model_save = f"{policy_model_save}-{num_agents}agents-{episode_length}months"
    if not os.path.exists(f"{save_path}data/{policy_model_save}"):
        os.makedirs(f"{save_path}data/{policy_model_save}")
    if not os.path.exists(f"{save_path}figs/{policy_model_save}"):
        os.makedirs(f"{save_path}figs/{policy_model_save}")
    for epi in range(env.episode_length):
        print(f"step {epi+1} start")
        actions = ollama_actions(
            env,
            obs,
            dialog_queue,
            dialog4ref_queue,
            f"{save_path}data/{policy_model_save}/dialogs",
        )

        obs, rew, done, info = env.step(actions)
        if (epi + 1) % 6 == 0 or epi + 1 == env.episode_length:
            with open(
                f"{save_path}data/{policy_model_save}/actions_{epi+1}.pkl", "wb"
            ) as f:
                pkl.dump(actions, f)
            with open(
                f"{save_path}data/{policy_model_save}/obs_{epi+1}.pkl", "wb"
            ) as f:
                pkl.dump(obs, f)
            with open(
                f"{save_path}data/{policy_model_save}/env_{epi+1}.pkl", "wb"
            ) as f:
                pkl.dump(env, f)
            if policy_model == "gpt":
                with open(
                    f"{save_path}data/{policy_model_save}/dialog_{epi+1}.pkl", "wb"
                ) as f:
                    pkl.dump(dialog_queue, f)
                with open(
                    f"{save_path}data/{policy_model_save}/dialog4ref_{epi+1}.pkl", "wb"
                ) as f:
                    pkl.dump(dialog4ref_queue, f)
            with open(
                f"{save_path}data/{policy_model_save}/dense_log_{epi+1}.pkl", "wb"
            ) as f:
                pkl.dump(env.dense_log, f)
        print(f"step {epi+1} done, cost {time()-t:.1f}s")
        t = time()

    with open(f"{save_path}data/{policy_model_save}/dense_log.pkl", "wb") as f:
        pkl.dump(env.dense_log, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate economic agents using Ollama.')
    parser.add_argument('--num_agents', type=int, default=20,
                        help='Number of agents in simulation (default: 20)')
    parser.add_argument('--episode_length', type=int, default=9,
                        help='Length of simulation episode in months (default: 9)')
    parser.add_argument('--dialog_len', type=int, default=3,
                        help='Maximum length of dialog history (default: 3)')
    parser.add_argument('--max_price_inflation', type=float, default=0.1,
                        help='Maximum price inflation rate (default: 0.1)')
    parser.add_argument('--max_wage_inflation', type=float, default=0.05,
                        help='Maximum wage inflation rate (default: 0.05)')
    
    args = parser.parse_args()
    
    main(
        num_agents=args.num_agents,
        episode_length=args.episode_length,
        dialog_len=args.dialog_len,
        max_price_inflation=args.max_price_inflation,
        max_wage_inflation=args.max_wage_inflation
    )