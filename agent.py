import json
import os

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Create a .env file containing GROQ_API_KEY=your_key"
    )

client = Groq(api_key=api_key)


# ============================================================
# 2. SESSION MEMORY
# ============================================================
# This memory survives across multiple calls to run_agent()
# during the same Python session.

memory = {
    "expenses": [],
    "monthly_budget": 5000.0,
}


# ============================================================
# 3. TOOL 1 - ADD EXPENSE
# ============================================================

def add_expense(
    item: str,
    amount: float,
    category: str = "general"
) -> str:
    """
    Add an expense to session memory.
    """

    expense = {
        "item": item,
        "amount": float(amount),
        "category": category,
    }

    memory["expenses"].append(expense)

    return (
        f"Expense successfully added: "
        f"{item} = ${amount:.2f}, category = {category}."
    )


# ============================================================
# 4. TOOL 2 - GET SUMMARY
# ============================================================

def get_summary(category: str = None) -> str:
    """
    Return spending information.

    If category is supplied, return spending for that category.
    Otherwise return the complete monthly budget summary.
    """

    # --------------------------------------------------------
    # Category-specific summary
    # --------------------------------------------------------

    if category:
        filtered = [
            expense
            for expense in memory["expenses"]
            if expense["category"].lower() == category.lower()
        ]

        category_total = sum(
            expense["amount"]
            for expense in filtered
        )

        return (
            f"Category: {category}\n"
            f"Total spent: ${category_total:.2f}\n"
            f"Transactions: {filtered}"
        )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    total_spent = sum(
        expense["amount"]
        for expense in memory["expenses"]
    )

    remaining = memory["monthly_budget"] - total_spent

    return (
        f"Monthly budget: ${memory['monthly_budget']:.2f}\n"
        f"Total spent: ${total_spent:.2f}\n"
        f"Remaining balance: ${remaining:.2f}\n"
        f"Expenses: {memory['expenses']}"
    )


# ============================================================
# 5. TOOL MAP
# ============================================================

tools_map = {
    "add_expense": add_expense,
    "get_summary": get_summary,
}


# ============================================================
# 6. TOOL SCHEMAS
# ============================================================

tools_schema = [

    # --------------------------------------------------------
    # add_expense
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "add_expense",
            "description": (
                "Record an expense in the user's session memory. "
                "Use this whenever the user tells you they spent money."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "Name or description of the expense."
                    },
                    "amount": {
                        "type": "number",
                        "description": "Amount spent."
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "Expense category such as food, travel, "
                            "study, shopping, or general."
                        )
                    }
                },
                "required": [
                    "item",
                    "amount"
                ]
            }
        }
    },

    # --------------------------------------------------------
    # get_summary
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "get_summary",
            "description": (
                "Retrieve the current spending and remaining monthly "
                "budget. Use this before answering questions about "
                "affordability or remaining money."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": (
                            "Optional category to calculate spending for."
                        )
                    }
                }
            }
        }
    }
]


# ============================================================
# 7. AGENT SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a Personal Budget Assistant agent.

Your job is to help a student track a monthly budget.

IMPORTANT AGENT RULES:

1. You are an agent, not just a chatbot.
2. Use tools to interact with the financial memory.
3. When the user reports an expense, use add_expense.
4. When the user asks about their balance, spending, or affordability,
   use get_summary before answering.
5. Never invent financial data.
6. For an affordability question such as:
   "Can I afford a $2000 weekend trip?"
   you must:
      a. Call get_summary to retrieve the current balance.
      b. Examine the returned balance.
      c. Calculate whether the proposed purchase fits.
      d. Decide whether the purchase is affordable.
      e. Explain the decision clearly.
7. Do not add a proposed purchase to the expense memory unless
   the user explicitly says they actually spent the money.
8. Remember information from earlier turns in this conversation.
9. Use previous tool results when making decisions later.
10. Show a concise explanation of the decision after completing
    the necessary tool calls.
"""


# ============================================================
# 8. AGENT ORCHESTRATION LOOP
# ============================================================

def run_agent(user_prompt: str, message_history: list) -> str:

    print("\n" + "=" * 60)
    print("USER PROMPT")
    print("=" * 60)

    print(user_prompt)

    # Add user message to conversation memory
    message_history.append({
        "role": "user",
        "content": user_prompt
    })

    # --------------------------------------------------------
    # PLAN -> ACT -> OBSERVE -> PLAN ...
    # --------------------------------------------------------

    while True:

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=message_history,
            tools=tools_schema,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        # ----------------------------------------------------
        # Store assistant response
        # ----------------------------------------------------

        message_history.append(msg)

        # ----------------------------------------------------
        # If the model doesn't need a tool, we're finished.
        # ----------------------------------------------------

        if not msg.tool_calls:

            print("\n" + "=" * 60)
            print("AGENT RESPONSE")
            print("=" * 60)

            print(msg.content)

            return msg.content

        # ----------------------------------------------------
        # Execute every requested tool
        # ----------------------------------------------------

        for tool_call in msg.tool_calls:

            function_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            print("\n" + "-" * 60)
            print("TRACE - TOOL CALL")
            print("-" * 60)

            print(
                f"{function_name}({arguments})"
            )

            # ------------------------------------------------
            # Safety check
            # ------------------------------------------------

            if function_name not in tools_map:
                raise ValueError(
                    f"Unknown tool requested: {function_name}"
                )

            # ------------------------------------------------
            # Execute tool
            # ------------------------------------------------

            tool_function = tools_map[function_name]

            tool_result = tool_function(**arguments)

            print("\nTRACE - TOOL RESULT")
            print("-" * 60)

            print(tool_result)

            # ------------------------------------------------
            # Send tool result back to model
            # ------------------------------------------------

            message_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_result)
            })

        # ----------------------------------------------------
        # Loop continues.
        #
        # The model now receives the tool result and decides
        # what to do next.
        # ----------------------------------------------------


# ============================================================
# 9. DEMO
# ============================================================

if __name__ == "__main__":

    conversation_history = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # --------------------------------------------------------
    # TURN 1
    # --------------------------------------------------------
    # The agent should call add_expense twice.
    # --------------------------------------------------------

    run_agent(
        "I bought textbooks for 1200 under study and spent "
        "300 on groceries.",
        conversation_history
    )

    # --------------------------------------------------------
    # TURN 2
    # --------------------------------------------------------
    # The agent should:
    #
    # 1. Remember the previous expenses.
    # 2. Call get_summary.
    # 3. Receive the current balance.
    # 4. Compare it with the proposed $2000 trip.
    # 5. Make a decision.
    # --------------------------------------------------------

    run_agent(
        "Can I afford a 2000 weekend trip right now?",
        conversation_history
    )