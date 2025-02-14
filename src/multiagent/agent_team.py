import sys
sys.path.append("../")

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import END, StateGraph, START


from pydantic import BaseModel, Field, field_validator
from typing import Annotated, Literal, Sequence, Dict, Any
from typing_extensions import TypedDict
import operator
import ast
import traceback
import functools

from multiagent.react_agent import create_react_agent
import multiagent.tools as tools
import multiagent.colors as colors
import json

class AgentTeam():

    def __init__(self, config, llm, member_list=[], member_names=[], member_info=[], team_name="Default", supervisor_name="FINISH",
                 intermediate_output_desc="", final_output_form="", fullshared_memory=False):
        self.config = config
        self.llm = llm
        self.graph = None

        self.member_list = member_list
        self.member_info = member_info
        self.member_names = member_names
        self.team_name = team_name
        self.sup_name = supervisor_name
        
        self.intermediate_output_desc = intermediate_output_desc
        self.final_output_form = final_output_form
        self.fullshared_memory = fullshared_memory

        self.message_prompt = f"If the next agent is one of {", ".join(member_names)}, give detailed instructions and requests. If {supervisor_name}, report a summary of all results."
        self.thought_prompt = "Output a detailed analysis on the most recent message. In detail, state what you think should be done next, and who you should contact next."
    


    def loadSuperviser(self, additional_prompt=""):
        options = [self.sup_name] + self.member_names
        job_list = "\n".join([mem + ": " + pr for mem, pr in zip(self.member_names, self.member_info)])
        prompt_list = [
                ("system",
                 "You are a supervisor tasked with managing a conversation between the following workers: {members}. "
                 "Given the following messages, respond with the worker to act next."
                 "Each worker will perform a task and respond with their results and status."
                 "\nJoblist: \n{joblist}"),
                
                ("system", "The current background is: {background}"),
                
                ("system", "Conversation History:"),
                
                MessagesPlaceholder(variable_name="messages", n_messages=20),
                
                ("system",
                 "Given the conversation above, output the following in this exact order:\n"
                 "1. 'thoughts': {thought_prompt}\n"
                 "2. 'messages': {message_prompt}\n"
                 "3. Who should act next? Select one of: {options} and output as 'next'."
                 " When you have determined that the final output is gained, report back with {finish} as 'next'.\n"
                 "4. The detailed background of the problem you are trying to solve (given in the first message) as 'background'.\n"
                 "5. The intermediate outputs to give as 'intermediate_output'.\n"
                 "" + additional_prompt + ""
                 "{error}"),
            ]
        if self.team_name == "Default":
            prompt_list.pop(1)
        
        prompt = ChatPromptTemplate.from_messages(prompt_list).partial(options=str(options), members=", ".join(self.member_names), joblist=job_list,\
                  finish=self.sup_name, sup_name=self.team_name + " Supervisor",\
                  thought_prompt=self.thought_prompt, message_prompt=self.message_prompt)



        class routeResponse(BaseModel):
            thoughts: str = Field(description=self.thought_prompt)
            next: Literal[*options] # type: ignore
            messages: str = Field(description=self.message_prompt)
            intermediate_output: str = Field(description=self.intermediate_output_desc)
            background: str
                
            @field_validator('messages', 'intermediate_output', mode="before")
            def cast_to_string(cls, v):
                return str(v)
            
        
        def supervisor_agent(state):
            prev_history = state["history"]

            if self.fullshared_memory:
                state["messages"] = list(state["history"]["all"])
            else:
                state["messages"] = list(state["history"][self.team_name + " Supervisor"])

            state["error"] = ""
            state["messages"] = state["messages"][-15:]
            state["history"] = {}

            # If this is the main team, there is no background.
            if self.team_name == "Default":
                state["background"] = ""
            else:
                state["background"] = state["background"].content

            supervisor_chain = prompt | self.llm.with_structured_output(routeResponse)

            for _ in range(10):
                try:
                    result = supervisor_chain.invoke(state)
                    break
                except Exception as e:
                    print(e)
                    print("Error occured. Retrying...")
                    state["error"] = "\n\nDouble check that 'next' is one of:  {options}."
            
            new_msg = AIMessage(content=result.messages, name=self.team_name + "_Supervisor")
            if not ("Intermediate Output" in new_msg.content or "Final Output" in new_msg.content):
                if result.intermediate_output in ["", "{{}}"]:
                    result.intermediate_output = state["intermediate_output"]
                if not str(result.intermediate_output) in new_msg.content:
                    new_msg.content = new_msg.content + "\n\nFinal Output: " + str(result.intermediate_output)

            if isinstance(result.intermediate_output, str):
                try:
                    result.intermediate_output = ast.literal_eval(result.intermediate_output)
                except Exception as e:
                    try:
                        result.intermediate_output = json.loads(result.intermediate_output)
                    except Exception as e2:
                        print("Could not parse.")# Current output:", result.intermediate_output)
                        result.intermediate_output = state["intermediate_output"]
                        print(e)
                        print(e2)
                if result.intermediate_output == {}:
                    result.intermediate_output = state["intermediate_output"]
            
            if self.fullshared_memory:
                new_history = {
                        **prev_history,
                        self.team_name + " Supervisor": prev_history.get(self.team_name + " Supervisor", []) + [new_msg.model_copy()],
                        result.next: prev_history.get(result.next, []) + [new_msg.model_copy()],
                        "all": prev_history.get("all", []) + [new_msg.model_copy()]
                    }
            else:
                new_history = {
                        **prev_history,
                        self.team_name + " Supervisor": prev_history.get(self.team_name + " Supervisor", []) + [new_msg.model_copy()],
                        result.next: prev_history.get(result.next, []) + [new_msg.model_copy()]
                    }
            
            new_history[self.team_name + " Supervisor"][-1].content = "{Thoughts: " + result.thoughts + "}\n\n" + new_history[self.team_name + " Supervisor"][-1].content
                    

            return {
                "intermediate_output": result.intermediate_output,
                "messages": new_msg,
                "background": AIMessage(content=result.background, name=self.team_name + " Supervisor"),
                "next": result.next,
                "history": new_history
            }
        

        return supervisor_agent
    

    def createStateGraph(self, additional_prompt=""):
            class AgentState(TypedDict):
                # history: Annotated[Sequence[BaseMessage], operator.add]
                history: Dict[str, Annotated[Sequence[BaseMessage], operator.add]]
                messages: BaseMessage
                background: BaseMessage
                intermediate_output: Dict[str, Any] = Field(description=self.intermediate_output_desc)
                next: str
                initial_prompt: BaseMessage

            if self.llm is None:
                self.loadLLM()
            workflow = StateGraph(AgentState)

            for member, member_name in zip(self.member_list, self.member_names):
                workflow.add_node(member_name, member)
            
            workflow.add_node(self.team_name + " Supervisor", self.loadSuperviser(additional_prompt=additional_prompt))
            for member in self.member_names:
                workflow.add_edge(member, self.team_name + " Supervisor")
            
            
            
            conditional_map = {k: k for k in self.member_names}
            conditional_map[self.sup_name] = END
            workflow.add_conditional_edges(self.team_name + " Supervisor", lambda x: x["next"], conditional_map)
            # Finally, add entrypoint
            workflow.add_edge(START, self.team_name + " Supervisor")

            self.graph = workflow.compile(debug = False)
            return functools.partial(AgentTeam.prompt, graph=self.graph, team_name=self.team_name, sup_name=self.sup_name if self.sup_name != "FINISH" else "", fullshared_memory=self.fullshared_memory)


    def prompt(state, config, graph, team_name, sup_name, fullshared_memory):
        int_output = {}
        color_assign = colors.TerminalColor()
        prev_len = len(state["history"][team_name + " Supervisor"])

        if "initial_prompt" in state:
            prev_initial_prompt = state["initial_prompt"]
            if not fullshared_memory:
                state["initial_prompt"] = "Prompter: " + state["history"][team_name + " Supervisor"][-1].name + "\n" + state["history"][team_name + " Supervisor"][-1].content
        else:
            prev_initial_prompt = state["history"][team_name + " Supervisor"][-1].content
            state["initial_prompt"] = prev_initial_prompt

        for s in graph.stream(
            state, config
        ):
            if "__end__" not in s:
                key = list(s.keys())[0]
                if key in ['history', 'messages', 'background', 'intermediate_output', 'next']:
                    result = s
                    try:
                        key = result['background'].name
                    except Exception as e:
                        print(e)
                        print(s["background"])
                        assert "Background error"
                    if prev_len == len(result["history"][team_name + " Supervisor"]):
                        continue
                else:
                    result = s[key]

                if int_output == {} and result["intermediate_output"] != {}:
                    int_output = result["intermediate_output"]
                print("Agent:", color_assign.colorText(key + f" ({len(result["history"][key])})", key))
                if key == team_name + " Supervisor":
                    print("Messages:")
                    print(color_assign.colorText(result["history"][key][-1].pretty_repr(), key))
                    print()

                if result["next"] != sup_name:
                    print("Background:")
                    print(color_assign.colorText(str(result["background"].content), key))
                    print()
                    if key != result["next"]:
                        print(color_assign.colorText(key, key), "->", color_assign.colorText(result["next"], result["next"]))
                    else:
                        print(color_assign.colorText(key, key), "->", color_assign.colorText(team_name + " Supervisor", team_name + " Supervisor"))
                    print()
        # print(s)
        state["initial_prompt"] = prev_initial_prompt
        if sup_name == "":
            return [result, int_output]
        return result
    


class ReactAgent():

    def __init__(self, config, intermediate_output_desc="", llm=None, key_type="GPT4", fullshared_memory=False):
        self.intermediate_output_desc = intermediate_output_desc
        self.config = config
        self.llm = llm

        self.key_type = key_type
        self.fullshared_memory = fullshared_memory

    def responseFormatter(result) -> dict:
        return {
                "intermediate_output": result.intermediate_output
            }


    def loadMember(self, name, sel_tools, member_prompt, sup_name):
        agent_tools = tools.getTools(sel_tools, self.config)


        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", member_prompt),
                ("system", "Background: {background}"),
                # ("system", "Incoming Instructions:\n{initial_prompt}"),
                ("system", "Conversation History:"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        class routeResponse(BaseModel):
            intermediate_output: str = Field(description=self.intermediate_output_desc)
                
            # @field_validator('intermediate_output', mode="before")
            # def cast_to_string(cls, v):
            #     return str(v)

        agent = create_react_agent(self.llm, tools=agent_tools, debug=False, state_modifier=prompt, response_schema=routeResponse, \
                                   response_format=ReactAgent.responseFormatter)
        return functools.partial(ReactAgent.agent_node, agent=agent, name=name, sup_name=sup_name, fullshared_memory=self.fullshared_memory)
    
    
    
    def agent_node(state, agent, name, sup_name="", fullshared_memory=False):
        color_assign = colors.TerminalColor()

        if fullshared_memory:
            state["messages"] = state["history"]["all"][-15:]
        else:
            state["messages"] = state["history"][name][-15:]
        prev_history = state["history"]
        state["history"] = {}

        prev_initial_prompt = state["initial_prompt"]
        
        if not fullshared_memory:
            state["initial_prompt"] = "Prompter: " + prev_history[name][-1].name + "\n" + prev_history[name][-1].content
        

        prev_len = len(state["messages"])

        for result in agent.stream(state, {}):
            if "__end__" not in result:
                
                if len(result["messages"]) != prev_len:
                    resp = result["messages"][-1]
                    if isinstance(resp, ToolMessage) and resp.name == "output_tool":
                        continue
                    # print(result)
                    try:
                        print("Agent:", color_assign.colorText(name + f' ({len(result["messages"])})', name))
                    except Exception as e:
                        print(e)
                        print(result)
                    print(color_assign.colorText(resp.pretty_repr(), name))
                    print()
                    prev_len = len(result["messages"])


        new_msg = AIMessage(content=result["messages"][-1].content, name=name.replace(" ", "_"))
        if not "Final Output" in new_msg.content and not result["intermediate_output"] in new_msg.content:
            new_msg.content = new_msg.content + "\n\nFinal Output: " + result["intermediate_output"]
        if fullshared_memory:
            result["history"] = {
                                    **prev_history,  # Keep the existing history
                                    sup_name: prev_history.get(sup_name, []) + [new_msg],
                                    name: prev_history.get(name, []) + [new_msg],
                                    "all": prev_history.get("all", []) + [new_msg]
                                }
        else:
            result["history"] = {
                                    **prev_history,  # Keep the existing history
                                    sup_name: prev_history.get(sup_name, []) + [new_msg],
                                    name: prev_history.get(name, []) + [new_msg]
                                }
            
        result["messages"] = new_msg
        result["next"] = state["next"]

        result["background"].name = name
        result["initial_prompt"] = prev_initial_prompt
        

        if isinstance(result["intermediate_output"], str):
            try:
                result["intermediate_output"] = ast.literal_eval(result["intermediate_output"])
            except Exception as e:
                try:
                    result["intermediate_output"] = json.loads(result["intermediate_output"])
                except Exception as e2:
                    print("Could not parse.")# Current output:", result["intermediate_output"])
                    result["intermediate_output"] = state["intermediate_output"]
                    print(e)
                    print(e2)

        return result





def buildTeam(team_information, react_generator, intermediate_output_desc, int_out_format):
    team_list = []
    member_info = []
    member_names = []

    for key in team_information.keys():
        if isinstance(team_information[key], dict):
            if "team" in team_information[key]:
                team_list.append(buildTeam(team_information[key], react_generator, intermediate_output_desc, int_out_format))
                member_names.append(team_information[key]["team"] + " Supervisor")
                member_info.append(team_information[key]["prompt"])
            else:
                team_list.append(react_generator.loadMember(key, team_information[key]["tools"], team_information[key]["prompt"], team_information["team"] + " Supervisor"))
                member_names.append(key)
                member_info.append(team_information[key]["prompt"])

    agent_team = AgentTeam(None, member_list=team_list, member_info=member_info, member_names=member_names, llm=react_generator.llm, \
                            intermediate_output_desc=intermediate_output_desc, final_output_form=int_out_format, team_name=team_information["team"], supervisor_name=team_information["return"])\
                                .createStateGraph(additional_prompt=team_information["additional_prompt"])
    return agent_team