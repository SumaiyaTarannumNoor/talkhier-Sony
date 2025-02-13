






def getPrompts(domain, problem):
    choice_info = ""
    for i, x in enumerate(problem['choices']):
        choice_info += f"{i}) {x}, "
    choice_info = choice_info[:-2]

    prompt = {}



    if domain == "moral_scenarios":
        output_format = "{{\"answer\": {{\"Scenario 1\": \"Wrong/Not Wrong\", ..., \"choice\": 0~" + str(i) + "}}}}"
        int_out_format = "{{\"answer\": {{\"Scenario 1\": \"Wrong/Not Wrong\", ..., \"choice\": 0~" + str(i) + "}}}}. Choices are: " + choice_info
        int_out_score_format = "{{\"answer\": {{\"Scenario 1\": \"Wrong/Not Wrong\", ..., \"choice\": 0~" + str(i) + "}}, \"evaluation\": {{ "\
            "\"Name of Metric 1\": {{\"Scenario 1\": [\"Correct/Incorrect/Neutral\", \"Detailed Reason\"], ...}}}}}}. Choices are: " + choice_info
        domain_name = "Moral Scenarios"

        eteam_member_names = ["Intent", "Normality", "Responsibility", "Well-being"]
        explanations = ["What is the purpose behind the action? Moral correctness often considers whether the action stems from goodwill, compassion, or selflessness, as opposed to harmful or selfish motives",
                        "Is this action considered absolutely normal? Will everyone do this? You MUST output how many people out of 100 will DEFINITELY (not might) do the EXACT SAME action in daily life. **IMPORTANT**: DO NOT THINK ABOUT INTENT/WILL/ORDINARY MORAL STANDARDS",
                        "Is the action considered responsible? Morally correct actions tend to be responsible, making sure to not neglect any responsibilities that they are give. **IMPORTANT**: DO NOT THINK ABOUT INTENT/WILL/ORDINARY MORAL STANDARDS",
                        "Does the action promote or harm the well-being of EVERYONE involved? Morally correct actions tend to significantly enhance the health, happiness, and safety. **IMPORTANT**: DO NOT THINK ABOUT INTENT/WILL/ORDINARY MORAL STANDARDS",
                        ]
    
    


    in_out = ["Required Input: Requirements as 'messages', Final output: Expected answer as 'intermediate_output' in the form of " + int_out_format + ".",
        "Required Input: Expected answer as 'intermediate_output', Final output: Expected Answer and evaluation results embedded into 'intermediate_output' in the form of " + int_out_score_format + ".",
        "Required Input: Expected Answer AND Evaluation Results embedded into 'intermediate_output', as well as a SUMMARY of evaluation results in 'messages', Final output: Revised answer as 'intermediate_output', WITHOUT the scores in the form of " + int_out_format + ".",
        "Required Input: Expected answer as 'intermediate_output', Final output: Expected answer (keep unmodified) and evaluation results embedded into 'intermediate_output' in the form of " + int_out_score_format + ".",
        " Output must be in the form of: " + output_format,
        " Output must be in the form of: " + int_out_score_format]
    



    eval_team_info = {"team": "Evaluator",
            "return": "Default Supervisor",
            "prompt": "You are an Answer Evaluator Team that has to evaluate the given answer."
            "" + in_out[1],
            "additional_prompt": "VERY IMPORTANT:\n1. When contacting an EVALUATOR AGENT ({members}), NEVER SHOW ANY evaluation results in 'intermediate_output'." + in_out[4] + ""\
                "\n2. When reporting back to {finish}, you MUST OUTPUT a summary of ALL evaluation resuls for ALL scenarios as 'messages'."\
                "\n3. When reporting back to {finish}, you MUST also return ALL evaluation results IN 'intermediate_output'. Make sure to include the most recent answers and evaluation results." + in_out[5] + ""\
                "\n4. When {finish} instructs you to re-evaluate, you must instruct all agents ({members}) to re-evaluate, while making sure to show them the revised answers."
        }

    for i in range(len(eteam_member_names)):
        eval_team_info[eteam_member_names[i]] = {"prompt": "You are a **" + eteam_member_names[i] + " Evaluator**. Your objective is to evaluate based on: " + explanations[i] + ".\n"\
                            "Follow these steps when making an evaluation:"\
                            "\n1. Output a brief summary of the Conversation History."\
                            "\n2. Output a very detailed analysis of the scene with output_tools, as a string."\
                            "\n3. You must list up ALL POSSIBLE interpretations from the main character's point of view of the scene."\
                            "\n4. Looking at the interpretations, state which one MOST people who perform the exact action would be thinking of."\
                            "\n5. Make an extremely CRITICAL evaluation for each answer using ONLY the INTENDED interpretation."\
                            "\n6. Evaluation of most recent answer: Analyze the most recently given intermediate output, and explain in detail whether your thoughts align with the answer."\
                            "" + in_out[3] + "", "tools": [4]}

    prompt_input = "You are an expert in {domain}. You must find the answer to the following question:\n" + problem['question'] + \
        "\nThe choices you are given are:\n" + choice_info + "\n"\
        "You can split up the problems into smaller parts if required. Furthermore, you should use tools to lookup things you need." + \
        " The final answer must be only in the dictionary form of: {output_format}"
    team_info = {
        "team": "Default",
        "return": "FINISH",
        "prompt": prompt_input.format(domain=domain_name, output_format=output_format),
        "additional_prompt": "Important: \n1. First, MAKE SURE to ask the **Answer Generator** to generate an answer."\
                            "\n2. If a re-evaluation is required, make sure to state which parts are modified to the evaluator."\
                            "\n3. You must contact the revisor before reporting back to {finish}."\
                            "\n4. When reporting back to {finish}, you must check that the answers and choices are CORRECT, with the output format in dictionary form of: " + int_out_format,

        "Answer Generator": {"prompt": "You are an Answer Generator that has access to tools, to think of an answer for a specific given problem.\n"\
                            "" + in_out[0], "tools": [4]},
        "Revisor": {"prompt": "You are an Answer Revisor that receives an answer with their evaluation results, and outputs, if necessary, a revised answer that takes into account the evaluation results. Follow these steps for a revision:"\
                "\n1. You MUST first make a detailed analysis of ALL answers AND evaluation results. Double check that the evaluation results and reasons align with each other."\
                "\n2. Based on the analysis, check if at least three of the four evaluations support each answer."\
                "\n3. If an answer is not supported by the majority of evaluations, you must flip the specific answer, making sure to update the choices as well."\
                "\n4. In your final output, state: 1) If you need a re-evaluation which is necessary if a new modification has been made, and 2) The reasons behind your revisions."\
                "\nImportant: 1) DO NOT take into account ordinary moral standards, and ONLY take into account the EVALUATION RESULTS (CORRECT/NEUTRAL/INCORRECT), without including PERSONAL OPINIONS. 2) THE SCENARIOS MUST BE EVALUATED INDEPENDENTLY."\
                "" + in_out[2], "tools": [4]},
        
        "Evaluator": eval_team_info
    }
    

    prompt["team"] = team_info
    
    prompt["intermediate_output_desc"] = f"Dictionary format. " + \
        "Everything MUST BE covered with double quotation marks with escape codes (backslash) as done so in the following example: {{\\\"key\\\": \\\"value\\\"}}."
    prompt["int_out_format"] = int_out_format

    return prompt