# CSE476 CA1 — Personal Budget Assistant Agent

## 1. Tools
This project implements a Personal Budget Assistant agent with two core tools: `add_expense(item, amount, category)`, which records a student's expense in session memory, and `get_summary(category)`, which retrieves either category spending or the overall monthly budget, total spending, and remaining balance. The agent uses these tools through a plan-act loop rather than simply generating a text response.

## 2. Memory
The agent maintains session memory containing the monthly budget and every expense recorded during the conversation. This memory is reused across turns. For example, after recording $1,200 for textbooks and $300 for groceries, the agent can later retrieve the $3,500 remaining balance when asked whether a $2,000 weekend trip is affordable. The memory is session-based and remains available while the Python process is running.

## 3. Failure and Handling
During development, the initial Groq model configuration using `llama-3.3-70b-versatile` returned a 404 `model_not_found` error because that model was not available to the API key/project being used. The issue was handled by checking the models available to the account and updating the agent to use `openai/gpt-oss-20b`, after which the tool-calling agent ran successfully. The API key is stored in a local `.env` file and is excluded from submission through `.gitignore`.

## Demo
Run the notebook `demo.ipynb` to see:
1. Multiple `add_expense` tool calls for a single user request.
2. An affordability decision using `get_summary()` and information stored from the previous turn.
3. A category spending query using `get_summary(category="study")`.
4. A final inspection of the agent's session memory.

## Project Structure

```text
INT476_CA1/
├── agent.py
├── demo.ipynb
├── README.md
├── .gitignore
└── .env                
