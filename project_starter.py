import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional
from sqlalchemy import create_engine, Engine

# Cargar variables de entorno
dotenv.load_dotenv()

# --- SMOLAGENTS IMPORTS ---
from smolagents import CodeAgent, Tool, OpenAIServerModel

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    {"item_name": "A4 paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                       "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                   "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                    "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                     "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                  "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",              "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                    "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                    "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                     "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",              "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                  "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                   "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                     "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                  "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                   "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",               "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",             "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",            "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                 "category": "paper",        "unit_price": 0.15},
    {"item_name": "Paper plates",                    "category": "product",      "unit_price": 0.10},
    {"item_name": "Paper cups",                      "category": "product",      "unit_price": 0.08},
    {"item_name": "Paper napkins",                   "category": "product",      "unit_price": 0.02},
    {"item_name": "Disposable cups",                 "category": "product",      "unit_price": 0.10},
    {"item_name": "Table covers",                    "category": "product",      "unit_price": 1.50},
    {"item_name": "Envelopes",                       "category": "product",      "unit_price": 0.05},
    {"item_name": "Sticky notes",                    "category": "product",      "unit_price": 0.03},
    {"item_name": "Notepads",                        "category": "product",      "unit_price": 2.00},
    {"item_name": "Invitation cards",                "category": "product",      "unit_price": 0.50},
    {"item_name": "Flyers",                          "category": "product",      "unit_price": 0.15},
    {"item_name": "Party streamers",                 "category": "product",      "unit_price": 0.05},
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},
    {"item_name": "Paper party bags",                "category": "product",      "unit_price": 0.25},
    {"item_name": "Name tags with lanyards",         "category": "product",      "unit_price": 0.75},
    {"item_name": "Presentation folders",            "category": "product",      "unit_price": 0.50},
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},
    {"item_name": "100 lb cover stock",              "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",               "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",            "category": "specialty",    "unit_price": 0.35},
]

# --- UTILITY FUNCTIONS (Provided in Starter Code) ---

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    np.random.seed(seed)
    num_items = int(len(paper_supplies) * coverage)
    selected_indices = np.random.choice(range(len(paper_supplies)), size=num_items, replace=False)
    selected_items = [paper_supplies[i] for i in selected_indices]
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),
            "min_stock_level": np.random.randint(50, 150)
        })
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    try:
        transactions_schema = pd.DataFrame({
            "id": [], "item_name": [], "transaction_type": [], "units": [], "price": [], "transaction_date": [],
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)
        initial_date = datetime(2025, 1, 1).isoformat()
        
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)
        
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))
        quotes_df = quotes_df[["request_id", "total_amount", "quote_explanation", "order_date", "job_type", "order_size", "event_type"]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)
        
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)
        initial_transactions = []
        initial_transactions.append({"item_name": None, "transaction_type": "sales", "units": None, "price": 50000.0, "transaction_date": initial_date})
        for _, item in inventory_df.iterrows():
            initial_transactions.append({"item_name": item["item_name"], "transaction_type": "stock_orders", "units": item["current_stock"], "price": item["current_stock"] * item["unit_price"], "transaction_date": initial_date})
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)
        return db_engine
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(item_name: str, transaction_type: str, quantity: int, price: float, date: Union[str, datetime]) -> int:
    try:
        date_str = date.isoformat() if isinstance(date, datetime) else date
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")
        transaction = pd.DataFrame([{"item_name": item_name, "transaction_type": transaction_type, "units": quantity, "price": price, "transaction_date": date_str}])
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])
    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    query = """
        SELECT item_name, SUM(CASE WHEN transaction_type = 'stock_orders' THEN units WHEN transaction_type = 'sales' THEN -units ELSE 0 END) as stock
        FROM transactions WHERE item_name IS NOT NULL AND transaction_date <= :as_of_date GROUP BY item_name HAVING stock > 0
    """
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    if isinstance(as_of_date, datetime): as_of_date = as_of_date.isoformat()
    stock_query = """
        SELECT item_name, COALESCE(SUM(CASE WHEN transaction_type = 'stock_orders' THEN units WHEN transaction_type = 'sales' THEN -units ELSE 0 END), 0) AS current_stock
        FROM transactions WHERE item_name = :item_name AND transaction_date <= :as_of_date
    """
    return pd.read_sql(stock_query, db_engine, params={"item_name": item_name, "as_of_date": as_of_date})

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    try: input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except: input_date_dt = datetime.now()
    days = 0 if quantity <= 10 else 1 if quantity <= 100 else 4 if quantity <= 1000 else 7
    return (input_date_dt + timedelta(days=days)).strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    try:
        if isinstance(as_of_date, datetime): as_of_date = as_of_date.isoformat()
        transactions = pd.read_sql("SELECT * FROM transactions WHERE transaction_date <= :as_of_date", db_engine, params={"as_of_date": as_of_date})
        if not transactions.empty:
            return float(transactions.loc[transactions["transaction_type"] == "sales", "price"].sum() - transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum())
        return 0.0
    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0

def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    if isinstance(as_of_date, datetime): as_of_date = as_of_date.isoformat()
    cash = get_cash_balance(as_of_date)
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value
        inventory_summary.append({"item_name": item["item_name"], "stock": stock, "unit_price": item["unit_price"], "value": item_value})
    top_sales_query = "SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue FROM transactions WHERE transaction_type = 'sales' AND transaction_date <= :date GROUP BY item_name ORDER BY total_revenue DESC LIMIT 5"
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    return {"as_of_date": as_of_date, "cash_balance": cash, "inventory_value": inventory_value, "total_assets": cash + inventory_value, "inventory_summary": inventory_summary, "top_selling_products": top_sales.to_dict(orient="records")}

def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    conditions = []
    params = {}
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(f"(LOWER(qr.response) LIKE :{param_name} OR LOWER(q.quote_explanation) LIKE :{param_name})")
        params[param_name] = f"%{term.lower()}%"
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    query = f"""
        SELECT qr.response AS original_request, q.total_amount, q.quote_explanation, q.job_type, q.order_size, q.event_type, q.order_date
        FROM quotes q JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause} ORDER BY q.order_date DESC LIMIT {limit}
    """
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
# YOUR MULTI AGENT STARTS HERE
########################

# 1. Define the Model
model = OpenAIServerModel(
    model_id="gpt-4-turbo",  
    api_base="https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY") 
)

# 2. Define Tools

class CheckStockTool(Tool):
    name = "check_stock"
    description = "Checks the current inventory level for a specific item name at a given date. Returns the count."
    inputs = {
        "item_name": {"type": "string", "description": "The exact name of the item (e.g., 'A4 paper')."},
        "date": {"type": "string", "description": "The date to check stock for (YYYY-MM-DD)."}
    }
    output_type = "integer"

    def forward(self, item_name: str, date: str) -> int:
        df = get_stock_level(item_name, date)
        if df.empty: return 0
        return int(df.iloc[0]["current_stock"])

class GetSupplierDeliveryTool(Tool):
    name = "get_delivery_estimate"
    description = "Estimates the delivery date from the supplier based on order quantity and start date."
    inputs = {
        "start_date": {"type": "string", "description": "The date ordering the items (YYYY-MM-DD)."},
        "quantity": {"type": "integer", "description": "The number of units to order."}
    }
    output_type = "string"

    def forward(self, start_date: str, quantity: int) -> str:
        return get_supplier_delivery_date(start_date, quantity)

class SearchHistoryTool(Tool):
    name = "search_history"
    description = "Searches historical quotes for similar requests to help determine pricing and discounts."
    inputs = {
        "terms": {"type": "string", "description": "Comma-separated search terms (e.g., 'large order, discount')."}
    }
    output_type = "string"

    def forward(self, terms: str) -> str:
        term_list = [t.strip() for t in terms.split(",")]
        results = search_quote_history(term_list)
        return str(results)

class GetItemPriceTool(Tool):
    name = "get_item_price"
    description = "Retrieves the base unit price and category for a specific item from the catalog."
    inputs = {
        "item_name": {"type": "string", "description": "The exact item name."}
    }
    output_type = "string"

    def forward(self, item_name: str) -> str:
        # Busca en la lista global paper_supplies
        found = next((item for item in paper_supplies if item["item_name"].lower() == item_name.lower()), None)
        if found:
            return str(found)
        return "Item not found in catalog."

class CreateTransactionTool(Tool):
    name = "create_db_transaction"
    description = "Records a sale or stock order in the database. Use this to finalize a sale."
    inputs = {
        "item_name": {"type": "string", "description": "Item name."},
        "trans_type": {"type": "string", "description": "'sales' or 'stock_orders'."},
        "qty": {"type": "integer", "description": "Quantity."},
        "total_price": {"type": "number", "description": "Total USD amount."},
        "date": {"type": "string", "description": "YYYY-MM-DD."}
    }
    output_type = "string"

    def forward(self, item_name: str, trans_type: str, qty: int, total_price: float, date: str) -> str:
        try:
            tid = create_transaction(item_name, trans_type, qty, total_price, date)
            return f"Transaction {tid} created successfully."
        except Exception as e:
            return f"Error creating transaction: {e}"

class CheckBalanceTool(Tool):
    name = "check_cash"
    description = "Checks the company's current cash balance."
    inputs = {
        "date": {"type": "string", "description": "YYYY-MM-DD."}
    }
    output_type = "number"

    def forward(self, date: str) -> float:
        return get_cash_balance(date)

# --- NEW HELPER CLASS FOR AGENT WRAPPING ---
class AgentTool(Tool):
    """
    Wrapper to allow an Agent to be used as a Tool by another Agent.
    Allows injecting 'system_instructions' into the call loop.
    """
    def __init__(self, agent, name, description, system_instructions=""):
        self.name = name
        self.description = description
        self.agent = agent
        self.system_instructions = system_instructions
        # Fix for "AttributeError: 'AgentTool' object has no attribute 'is_initialized'"
        self.is_initialized = True 
        
        # Define strict input schema for the orchestrator to know how to call it
        self.inputs = {
            "request": {
                "type": "string", 
                "description": "The natural language instruction/request for this agent."
            }
        }
        self.output_type = "string"

    def forward(self, request: str) -> str:
        # Prepend instructions to the request to enforce the persona
        full_prompt = f"{self.system_instructions}\n\nTask Request: {request}"
        return self.agent.run(full_prompt)

# 3. Instantiate Agents

# NOTE: Removed 'system_prompt', 'name', and 'description' from CodeAgent constructor 
# to avoid compatibility issues. Instructions are passed via AgentTool wrapper.

# -- Inventory Agent --
inventory_agent_core = CodeAgent(
    tools=[CheckStockTool(), GetSupplierDeliveryTool()],
    model=model
)

# -- Quoting Agent --
quoting_agent_core = CodeAgent(
    tools=[SearchHistoryTool(), GetItemPriceTool()],
    model=model
)

# -- Sales Agent --
sales_agent_core = CodeAgent(
    tools=[CreateTransactionTool(), CheckBalanceTool()],
    model=model
)

# -- WRAP AGENTS AS TOOLS (This is where we define the persona) --

inventory_tool = AgentTool(
    inventory_agent_core, 
    "inventory_tool", 
    "Check stock and delivery dates. Input: A request like 'Check stock for A4 paper'.",
    system_instructions="""You are the Inventory Manager. 
    1. Always check stock levels when asked about an item availability.
    2. If stock is insufficient, use the delivery estimate tool to say when more can arrive.
    3. Return a clear summary of stock status."""
)

quoting_tool = AgentTool(
    quoting_agent_core, 
    "quoting_tool", 
    "Calculate prices and quotes. Input: A request like 'Quote for 500 sheets of A4'.",
    system_instructions="""You are the Quoting Specialist.
    1. Identify items in the user request.
    2. Get their base price using get_item_price.
    3. Search history for similar previous orders to see if discounts apply.
    4. Calculate the final total price.
    5. Return the price per item and the Grand Total."""
)

sales_tool = AgentTool(
    sales_agent_core, 
    "sales_tool", 
    "Finalize sales and book transactions. Input: 'Sell 500 A4 paper for $25'.",
    system_instructions="""You are the Sales Manager.
    1. Your goal is to finalize the deal.
    2. Record the transaction using 'create_db_transaction' for EACH item sold.
    3. Transaction type is 'sales'.
    4. Return a confirmation message."""
)

# -- Orchestrator Agent --
# The orchestrator will use the wrapped tools to communicate with the sub-agents.
orchestrator_agent = CodeAgent(
    tools=[inventory_tool, quoting_tool, sales_tool], 
    model=model
)

# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    print("Initializing Database...")
    init_database(db_engine)
    
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        # Ensure dates are parsed correctly if they exist, or fallback
        if "request_date" in quote_requests_sample.columns:
             # Try parsing mixed formats if necessary, assuming MM/DD/YY based on sample
            quote_requests_sample["request_date"] = pd.to_datetime(
                quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
            )
        else:
             # Fallback if column missing
             quote_requests_sample["request_date"] = datetime(2025, 4, 1)

        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]
    
    results = []

    print("\nStarting Multi-Agent Simulation...")
    print("-" * 50)

    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Date: {request_date}")
        print(f"Request: {row['request']}")
        
        # Prepare the prompt for the Orchestrator
        # We pass the persona instructions here in the run loop prompt
        prompt = f"""
        You are the Chief Operator at Munder Difflin.
        Current Date: {request_date}
        Customer Request: "{row['request']}"
        
        Workflow:
        1. Check inventory using 'inventory_tool'.
        2. Calculate the price using 'quoting_tool'.
        3. If successful, record the sale using 'sales_tool'.
        
        CRITICAL: Pass the date '{request_date}' to any tool that needs it.
        """

        try:
            # Run the agent
            response = orchestrator_agent.run(prompt)
        except Exception as e:
            response = f"Agent Error: {str(e)}"
            print(response)

        # Update state to see changes
        report = generate_financial_report(request_date)
        new_cash = report["cash_balance"]
        new_inventory = report["inventory_value"]
        
        print(f"Agent Response: {response}")
        print(f"Cash Change: ${new_cash - current_cash:.2f}")
        
        results.append({
            "request_id": idx + 1,
            "request_date": request_date,
            "cash_balance": new_cash,
            "inventory_value": new_inventory,
            "response": str(response),
            "cash_change": new_cash - current_cash
        })
        
        current_cash = new_cash
        current_inventory = new_inventory
        time.sleep(1) # Pause purely for readability if watching stdout

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    print("Results saved to test_results.csv")
    return results

if __name__ == "__main__":
    run_test_scenarios()