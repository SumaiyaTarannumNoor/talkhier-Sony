from typing import Callable, Literal, Optional, Sequence, Type, TypeVar, Union, cast

from langchain_core.language_models import BaseChatModel, LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import (
    Runnable,
    RunnableBinding,
    RunnableConfig,
)
from langchain_core.tools import BaseTool
from typing_extensions import Annotated, TypedDict

from langgraph._api.deprecation import deprecated_parameter
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer
from langgraph.utils.runnable import RunnableCallable


# We create the AgentState that we will pass around
# This simply involves a list of messages
# We want steps to return messages to append to the list
# So we annotate the messages attribute with operator.add


from pydantic import BaseModel

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    background: BaseMessage
    intermediate_output: str
    is_last_step: IsLastStep
    initial_prompt: BaseMessage



StateSchema = TypeVar("StateSchema", bound=AgentState)
StateSchemaType = Type[StateSchema]

STATE_MODIFIER_RUNNABLE_NAME = "StateModifier"

MessagesModifier = Union[
    SystemMessage,
    str,
    Callable[[Sequence[BaseMessage]], Sequence[BaseMessage]],
    Runnable[Sequence[BaseMessage], Sequence[BaseMessage]],
]

StateModifier = Union[
    SystemMessage,
    str,
    Callable[[StateSchema], Sequence[BaseMessage]],
    Runnable[StateSchema, Sequence[BaseMessage]],
]


def _get_state_modifier_runnable(
    state_modifier: Optional[StateModifier], store: Optional[BaseStore] = None
) -> Runnable:
    state_modifier_runnable: Runnable
    if state_modifier is None:
        state_modifier_runnable = RunnableCallable(
            lambda state: state["messages"], name=STATE_MODIFIER_RUNNABLE_NAME
        )
    elif isinstance(state_modifier, str):
        _system_message: BaseMessage = SystemMessage(content=state_modifier)
        state_modifier_runnable = RunnableCallable(
            lambda state: [_system_message] + state["messages"],
            name=STATE_MODIFIER_RUNNABLE_NAME,
        )
    elif isinstance(state_modifier, SystemMessage):
        state_modifier_runnable = RunnableCallable(
            lambda state: [state_modifier] + state["messages"],
            name=STATE_MODIFIER_RUNNABLE_NAME,
        )
    elif callable(state_modifier):
        state_modifier_runnable = RunnableCallable(
            state_modifier,
            name=STATE_MODIFIER_RUNNABLE_NAME,
        )
    elif isinstance(state_modifier, Runnable):
        state_modifier_runnable = state_modifier
    else:
        raise ValueError(
            f"Got unexpected type for `state_modifier`: {type(state_modifier)}"
        )

    return state_modifier_runnable


def _convert_messages_modifier_to_state_modifier(
    messages_modifier: MessagesModifier,
) -> StateModifier:
    state_modifier: StateModifier
    if isinstance(messages_modifier, (str, SystemMessage)):
        return messages_modifier
    elif callable(messages_modifier):

        def state_modifier(state: AgentState) -> Sequence[BaseMessage]:
            return messages_modifier(state["messages"])

        return state_modifier
    elif isinstance(messages_modifier, Runnable):
        state_modifier = (lambda state: state["messages"]) | messages_modifier
        return state_modifier
    raise ValueError(
        f"Got unexpected type for `messages_modifier`: {type(messages_modifier)}"
    )


def _get_model_preprocessing_runnable(
    state_modifier: Optional[StateModifier],
    messages_modifier: Optional[MessagesModifier],
    store: Optional[BaseStore],
) -> Runnable:
    # Add the state or message modifier, if exists
    if state_modifier is not None and messages_modifier is not None:
        raise ValueError(
            "Expected value for either state_modifier or messages_modifier, got values for both"
        )

    if state_modifier is None and messages_modifier is not None:
        state_modifier = _convert_messages_modifier_to_state_modifier(messages_modifier)

    return _get_state_modifier_runnable(state_modifier, store)


def _should_bind_tools(model: LanguageModelLike, tools: Sequence[BaseTool]) -> bool:
    if not isinstance(model, RunnableBinding):
        return True

    if "tools" not in model.kwargs:
        return True

    bound_tools = model.kwargs["tools"]
    if len(tools) != len(bound_tools):
        raise ValueError(
            "Number of tools in the model.bind_tools() and tools passed to create_react_agent must match"
        )

    tool_names = set(tool.name for tool in tools)
    bound_tool_names = set()
    for bound_tool in bound_tools:
        # OpenAI-style tool
        if bound_tool.get("type") == "function":
            bound_tool_name = bound_tool["function"]["name"]
        # Anthropic-style tool
        elif bound_tool.get("name"):
            bound_tool_name = bound_tool["name"]
        else:
            # unknown tool type so we'll ignore it
            continue

        bound_tool_names.add(bound_tool_name)

    if missing_tools := tool_names - bound_tool_names:
        raise ValueError(f"Missing tools '{missing_tools}' in the model.bind_tools()")

    return False




from pydantic import BaseModel

def create_react_agent(
    model: LanguageModelLike,
    tools: Union[ToolExecutor, Sequence[BaseTool], ToolNode],
    response_schema: Optional[BaseModel],
    response_format,
    *,
    state_schema: Optional[StateSchemaType] = None,
    messages_modifier: Optional[MessagesModifier] = None,
    state_modifier: Optional[StateModifier] = None,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
    debug: bool = False,
) -> CompiledGraph:

    if state_schema is not None:
        if missing_keys := {"messages", "is_last_step"} - set(
            state_schema.__annotations__
        ):
            raise ValueError(f"Missing required key(s) {missing_keys} in state_schema")

    if isinstance(tools, ToolExecutor):
        tool_classes: Sequence[BaseTool] = tools.tools
        tool_node = ToolNode(tool_classes)
    elif isinstance(tools, ToolNode):
        tool_classes = list(tools.tools_by_name.values())
        tool_node = tools
    else:
        tool_node = ToolNode(tools)
        # get the tool functions wrapped in a tool class from the ToolNode
        tool_classes = list(tool_node.tools_by_name.values())

    if _should_bind_tools(model, tool_classes):
        model = cast(BaseChatModel, model).bind_tools(tool_classes)

    # Define the function that determines whether to continue or not
    if response_schema is not None:
        def should_continue(state: AgentState) -> Literal["tools", "respond"]:
            messages = state["messages"]
            last_message = messages[-1]
            # If there is no function call, then we finish
            if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
                return "respond"
                # return "__end__"
            # Otherwise if there is, we continue
            else:
                return "tools"
    else:
        def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
            messages = state["messages"]
            last_message = messages[-1]
            # If there is no function call, then we finish
            if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
                return "__end__"
            # Otherwise if there is, we continue
            else:
                return "tools"

    # we're passing store here for validation
    preprocessor = _get_model_preprocessing_runnable(
        state_modifier, messages_modifier, store
    )
    

    if response_schema is not None:
        model_response_runnable = preprocessor | model.with_structured_output(response_schema)
        def final_response(state: AgentState, config: RunnableConfig) -> AgentState:
            for _ in range(10):
                try:
                    result =  model_response_runnable.invoke(state, config)
                    return response_format(result)
                except Exception as e:
                    print(e)
                    print("Error occured. Retrying...")
                    state["messages"].append("Error message: " + str(e))
            


    model_runnable = preprocessor | model


    # Define the function that calls the model
    def call_model(state: AgentState, config: RunnableConfig) -> AgentState:
        response = model_runnable.invoke(state, config)
        
        if (
            state["is_last_step"]
            and isinstance(response, AIMessage)
            and response.tool_calls
        ):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
        response = await model_runnable.ainvoke(state, config)
        if (
            state["is_last_step"]
            and isinstance(response, AIMessage)
            and response.tool_calls
        ):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    # Define a new graph
    workflow = StateGraph(state_schema or AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", RunnableCallable(call_model, acall_model))
    workflow.add_node("tools", tool_node)

    if response_schema is not None:
        workflow.add_node("respond", final_response)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
    )

    # If any of the tools are configured to return_directly after running,
    # our graph needs to check if these were called
    should_return_direct = {t.name for t in tool_classes if t.return_direct}

    def route_tool_responses(state: AgentState) -> Literal["agent", "__end__"]:
        for m in reversed(state["messages"]):
            if not isinstance(m, ToolMessage):
                break
            if m.name in should_return_direct:
                return "__end__"
        return "agent"

    if should_return_direct:
        workflow.add_conditional_edges("tools", route_tool_responses)
    else:
        workflow.add_edge("tools", "agent")
        if response_schema is not None:
            workflow.add_edge("respond", "__end__")

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    return workflow.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
    )


# Keep for backwards compatibility
create_tool_calling_executor = create_react_agent

__all__ = [
    "create_react_agent",
    "create_tool_calling_executor",
    "AgentState",
]