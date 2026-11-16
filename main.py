from __future__ import annotations

import json
import os
import random
import re
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

# Starter inventory; replace with the restaurant's actual menu and stock.
MENU: dict[str, int] = {
    "margherita pizza": 8,
    "veggie burger": 5,
    "pasta": 6,
}


def message_content(message: object) -> str:
    """Read content from either a LangChain message or a plain message dict."""
    if hasattr(message, "content"):
        return str(message.content)
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(message)


class OrderState(TypedDict, total=False):
    messages: Annotated[list[dict[str, str]], add_messages]
    dish_name: str
    required_quantity: int
    available_quantity: str
    status: str
    # Number of new order submissions allowed after the initial submission.
    order_retry_attempts: int
    cook_retry_attempts: int
    serve_retry_attempts: int
    cook_started: bool
    final_result: str
    last_error: str


def _llm_extract(text: str) -> dict[str, object]:
    """Extract one food order with an optional LLM; use a local parser without a key."""
    if os.getenv("GROQ_API_KEY"):
        try:
            from langchain_groq import ChatGroq

            model = ChatGroq(
                model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                temperature=0,
            ).bind(
                response_format={"type": "json_object"}
            )
            response = model.invoke(
                [
                    (
                        "system",
                        "You are the order parser for a restaurant ordering system. "
                        "Extract only one dish and a positive integer quantity. Return JSON "
                        "with is_food_order (boolean), dish (string), and quantity (integer). "
                        "For unrelated, unclear, or multiple-dish requests, return "
                        '{"is_food_order": false, "dish": "", "quantity": 0}.',
                    ),
                    ("human", text),
                ]
            )
            data = json.loads(str(response.content))
            quantity = int(data.get("quantity", 0))
            dish = str(data.get("dish", "")).strip()
            return {
                "is_food_order": bool(data.get("is_food_order")) and bool(dish) and quantity > 0,
                "dish": dish,
                "quantity": quantity,
            }
        except Exception:
            # If the model is unavailable, allow the basic CLI parser to keep working.
            pass

    match = re.fullmatch(r"\s*(\d+)\s+(.+?)\s*", text)
    if match:
        quantity, dish = match.group(1), match.group(2)
    else:
        match = re.fullmatch(r"\s*(.+?)\s+(\d+)\s*", text)
        if not match:
            return {"is_food_order": False, "dish": "", "quantity": 0}
        dish, quantity = match.group(1), match.group(2)

    dish = re.sub(
        r"^(?:i(?:'d)?\s+like|order|please|can i have)\s+",
        "",
        dish,
        flags=re.IGNORECASE,
    )
    dish = re.sub(r"\s+(?:please|thanks?)$", "", dish, flags=re.IGNORECASE).strip()
    return {
        "is_food_order": bool(dish and int(quantity) > 0),
        "dish": dish,
        "quantity": int(quantity),
    }


def extract_order(state: OrderState) -> dict:
    # ask_order appends the new user submission last in the annotated messages list.
    order = _llm_extract(message_content(state["messages"][-1]))
    if not order["is_food_order"]:
        return {"status": "unsupported"}
    return {
        "dish_name": str(order["dish"]).strip().lower(),
        "required_quantity": int(order["quantity"]),
        "status": "pending_confirmation",
        "available_quantity": "0",
    }


def order_confirm(state: OrderState) -> dict:
    dish = state["dish_name"].strip().lower()
    available = next((count for name, count in MENU.items() if name.lower() == dish), 0)
    requested = state["required_quantity"]
    if available <= 0:
        status = "unavailable"
    elif available < requested:
        status = "partial"
    else:
        status = "confirmed"
    updates = {"available_quantity": str(available), "status": status}
    if status != "confirmed":
        # A partial/unavailable result uses one order retry attempt.
        updates["order_retry_attempts"] = state["order_retry_attempts"] - 1
    return updates


def order_decision(state: OrderState) -> Literal["cook", "ask_order", "finish"]:
    if state["status"] == "confirmed":
        return "cook"
    if state["order_retry_attempts"] > 0:
        return "ask_order"
    return "finish"


def ask_order(state: OrderState) -> dict:
    available = int(state.get("available_quantity", "0"))
    if state["status"] == "partial":
        prompt = (
            f"Only {available} of {state['dish_name']} are available. "
            f"Type 'accept' to order {available}, or enter a new dish and quantity."
        )
    else:
        prompt = f"{state['dish_name']} is unavailable. Enter a new dish and quantity."

    answer = input(f"Assistant: {prompt}\nYou: ").strip()
    updates: dict = {
        "messages": [
            {"role": "assistant", "content": prompt},
            {"role": "user", "content": answer},
        ],
    }
    if answer.lower() in {"accept", "yes", "confirm"} and state["status"] == "partial":
        updates.update({"required_quantity": available, "status": "confirmed"})
    else:
        # Choosing another order consumes a retry too.
        updates["order_retry_attempts"] = state["order_retry_attempts"] - 1
    return updates


def route_after_ask(state: OrderState) -> Literal["cook", "extract_order", "finish"]:
    if state["status"] == "confirmed":
        return "cook"
    if state["order_retry_attempts"] <= 0:
        return "finish"
    return "extract_order"


def cook(state: OrderState) -> dict:
    retries_remaining = state["cook_retry_attempts"]
    if state.get("cook_started", False):
        # Every repeated cook consumes a retry, including a successful one.
        retries_remaining = max(0, retries_remaining - 1)
    if random.random() < 0.60:
        return {
            "status": "ready",
            "cook_retry_attempts": retries_remaining,
            "cook_started": True,
        }
    return {
        "status": "cook_failed",
        "cook_retry_attempts": retries_remaining,
        "cook_started": True,
        "last_error": "Cooking failed.",
    }


def route_after_cook(state: OrderState) -> Literal["serve", "cook", "finish"]:
    if state["status"] == "ready":
        return "serve"
    if state["cook_retry_attempts"] > 0:
        return "cook"
    return "finish"


def serve(state: OrderState) -> dict:
    if random.random() < 0.60:
        return {"status": "complete", "final_result": "completed"}
    return {
        "status": "serve_failed",
        "serve_retry_attempts": state["serve_retry_attempts"] - 1,
        "last_error": "Serving failed.",
    }


def route_after_serve(state: OrderState) -> Literal["finish", "cook"]:
    if state["status"] == "complete":
        return "finish"
    # A failed serve uses one remaining serve attempt and requires another cook.
    if state["serve_retry_attempts"] <= 0 or state["cook_retry_attempts"] <= 0:
        return "finish"
    return "cook"


def finish(state: OrderState) -> dict:
    if state.get("status") == "complete":
        message = (
            f"Your order of {state['required_quantity']} "
            f"{state['dish_name']} is complete!"
        )
        result = "completed"
    elif state.get("status") == "unsupported":
        message = (
            "I'm a food-ordering assistant. Enter one dish and quantity "
            "(for example, '2 pasta')."
        )
        result = "not_completed"
    else:
        message = "I'm sorry, but I couldn't complete your order."
        result = "not_completed"
    return {
        "final_result": result,
        "messages": [{"role": "assistant", "content": message}],
    }


def build_graph():
    graph = StateGraph(OrderState)
    for name, node in [
        ("extract_order", extract_order),
        ("order_confirm", order_confirm),
        ("ask_order", ask_order),
        ("cook", cook),
        ("serve", serve),
        ("finish", finish),
    ]:
        graph.add_node(name, node)

    graph.add_edge(START, "extract_order")
    graph.add_conditional_edges(
        "extract_order",
        lambda state: (
            "order_confirm"
            if state["status"] == "pending_confirmation"
            else "finish"
        ),
        {"order_confirm": "order_confirm", "finish": "finish"},
    )
    graph.add_conditional_edges(
        "order_confirm",
        order_decision,
        {"cook": "cook", "ask_order": "ask_order", "finish": "finish"},
    )
    graph.add_conditional_edges(
        "ask_order",
        route_after_ask,
        {"cook": "cook", "extract_order": "extract_order", "finish": "finish"},
    )
    graph.add_conditional_edges(
        "cook",
        route_after_cook,
        {"serve": "serve", "cook": "cook", "finish": "finish"},
    )
    graph.add_conditional_edges(
        "serve",
        route_after_serve,
        {"finish": "finish", "cook": "cook"},
    )
    graph.add_edge("finish", END)
    return graph.compile()


def main() -> None:
    print(
        "Restaurant order assistant. Enter one dish and quantity "
        "(e.g. '2 pasta'). Type 'quit' to exit."
    )
    graph = build_graph()
    while True:
        text = input("You: ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        state: OrderState = {
            "messages": [{"role": "user", "content": text}],
            "order_retry_attempts": 3,
            "cook_retry_attempts": 2,
            "serve_retry_attempts": 2,
            "status": "new",
            "final_result": "not_completed",
        }
        result = graph.invoke(state)
        print(f"Assistant: {message_content(result['messages'][-1])}")


if __name__ == "__main__":
    main()



