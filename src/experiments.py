import configparser
import argparse
import shutil, os
import ast
import atexit

from result_manager import ResultManager
import multiagent.agent_team as agent_team
from multiagent.llm import LLM
import prompts.mmlu as MMLUPrompt



def mmlu(problem, prompt, config):
    choice_info = ""
    for i, x in enumerate(problem['choices']):
        choice_info += f"{i}) {x}, "
    choice_info = choice_info[:-2]

    react_generator = agent_team.ReactAgent(intermediate_output_desc=prompt["intermediate_output_desc"], config=config, llm=LLM(config).llm)
    team = agent_team.buildTeam(prompt["team"], react_generator, prompt["intermediate_output_desc"], prompt["int_out_format"])

    from langchain_core.messages import HumanMessage

    iterations = 0
    while iterations < 10:
        try:
            output = team(state = {"history": {"Default Supervisor": [HumanMessage(content=prompt["team"]["prompt"])], "all": [HumanMessage(content=prompt["team"]["prompt"])]}, "intermediate_output": {}},
            config = {"recursion_limit": int(config["LLM"]["max_attempts"])})
            break
        except Exception as e:
            print(e)
            print("Retrying...")


    if isinstance(output, list):
        output, output_int = output[0]["intermediate_output"], output[1]
    else:
        print(output)
        output = ast.literal_eval(output["messages"][-1].content)

    choice = -1
    if "choice" in output:
        choice = output["choice"]
    elif "choice" in output["answer"]:
        choice = output["answer"]["choice"]
    
    choice2 = -1
    if output_int is not None and output_int != {}:
        if "choice" in output_int:
            choice2 = output_int["choice"]
        elif "choice" in output_int["answer"]:
            choice2 = output_int["answer"]["choice"] 
    
    return choice, choice2






def runExperimentMMLU(file_path, domain, config):
    
    from datasets import load_dataset
    dataset = load_dataset("cais/mmlu", domain)
    results = ResultManager(file_path, columns=["Problem ID", "Answer", "Correct Answer", "Pre-Revision"])

    for problem_id in range(len(dataset["test"])):
        placeholder = [problem_id, '-', '-', '-']

        if not results.is_present(placeholder):

            def remove_checkpoint():
                results.remove(placeholder)

            results.add(placeholder)
            atexit.register(remove_checkpoint)

            problem = dataset["test"][problem_id]
            answer = mmlu(problem, MMLUPrompt.getPrompts(domain, problem), config)

            results.replace(placeholder, [problem_id, answer[0], problem["answer"], answer[1]])
            atexit.unregister(remove_checkpoint)



def runSingleMMLU(domain, config, problem_id):
    from datasets import load_dataset
    dataset = load_dataset("cais/mmlu", domain)
    problem = dataset["test"][problem_id]

    output = mmlu(problem, MMLUPrompt.getPrompts(domain, problem), config)
    print("Agent Answer:", output[0])
    print("Agent Answer before Revision:", output[1])
    print("Actual Answer:", problem["answer"])




def loadConfig(path, path2):
    config = configparser.ConfigParser()
    config.read(path, encoding='utf-8')

    cconfig = configparser.ConfigParser()
    cconfig.read(path2, encoding='utf-8')

    for section in cconfig.sections():
        if not config.has_section(section):
            config.add_section(section)
        for key, value in cconfig.items(section):
            config.set(section, key, value)

    return config

def getArgs():
    parser = argparse.ArgumentParser(description="Talk Structurally, Act Hierarchically (TalkHier): A Collaborative Framework for LLM Multi-Agent Systems")
    parser.add_argument('--config', '-cfg',
                        type=str,
                        default="../config/config.ini",
                        help="Path to config file.")
    parser.add_argument('--llm-config', '-lcfg', 
                        type=str, 
                        default="../config/config_llm.ini",
                        help="Path to LLM config file.")
    parser.add_argument('--mode', '-m', 
                        type=str, 
                        choices=["camera", "moral_scenarios", "college_physics", "machine_learning", "formal_logic", "us_foreign_policy", "wiki_qa"], 
                        default="camera",
                        help="Type of dataset to experiment on.")
    parser.add_argument('--problem-id', '-pi',
                        type=int,
                        default=-1,
                        help="Specific problem to execute; -1 for experiments on entire dataset.")
    parser.add_argument('--thread-count', '-tc', 
                        type=int, 
                        default=1,
                        help="Number of experiments to run in parallel.")
    parser.add_argument('--vanilla', '-v', 
                        action="store_true",
                        help="Option to run with unmodified GPT.")

    return parser.parse_args()




if __name__ == "__main__":
    args = getArgs()

    if not os.path.exists(args.llm_config):
        shutil.copy("../template/config_llm.ini", args.llm_config)
        print("Initialized config file.")
        print("Please write down the keys into:", args.llm_config)
        exit()

    if not os.path.exists(args.config):
        shutil.copy("../template/config.ini", args.config)

    config = loadConfig(args.config, args.llm_config)
    mmlu_list = ["moral_scenarios", "college_physics", "machine_learning", "formal_logic", "us_foreign_policy"]
    prefix = "proposed.csv"

    if args.problem_id != -1:
        if args.mode in mmlu_list:
            runSingleMMLU(args.mode, config, args.problem_id)
    else:
        if args.mode in mmlu_list:
            runExperimentMMLU("../results/" + args.mode + "/" + prefix, args.mode, config)