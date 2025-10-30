"""
Walkie Rental Manager - Streamlit App with OpenAI Chatbot
File: walkie_rental_app.py

Features:
- Inventory management with asset tracking component
- OpenAI-powered chatbot with database tool binding
- Add new booking (client, phone, units, start/end date, price/day)
- Active rentals list with due dates
- Mark returns (partial or full), record notes and damage
- Stats on total, rented, available, income
- Export bookings to CSV
- Persist data in a local SQLite database (walkie_rental.db)
- Real-time asset status visualization

Run:
1) pip install streamlit pandas python-dateutil openai
2) Set OPENAI_API_KEY environment variable
3) streamlit run walkie_rental_app.py

Author: Enhanced for Rivindu
"""

import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, date, timedelta
import uuid
from io import StringIO
import threading
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

DB_PATH = "walkie_rental.db"
DEFAULT_TOTAL = 50

# Thread lock for database operations
db_lock = threading.Lock()

@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db(conn):
    with db_lock:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS bookings (
                booking_id TEXT PRIMARY KEY,
                client_name TEXT,
                phone TEXT,
                units INTEGER,
                price_per_day REAL,
                start_date TEXT,
                end_date TEXT,
                created_at TEXT,
                returned_units INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active',
                notes TEXT
            )
        ''')
        conn.commit()
        # set default total walkies if not set
        c.execute("SELECT value FROM settings WHERE key='total_walkies'")
        row = c.fetchone()
        if not row:
            c.execute("INSERT INTO settings(key,value) VALUES(?,?)", ('total_walkies', str(DEFAULT_TOTAL)))
            conn.commit()

def get_total_walkies(conn):
    with db_lock:
        c = conn.cursor()
        c.execute("SELECT value FROM settings WHERE key='total_walkies'")
        row = c.fetchone()
        return int(row[0]) if row else DEFAULT_TOTAL

def set_total_walkies(conn, val):
    with db_lock:
        c = conn.cursor()
        c.execute("REPLACE INTO settings(key,value) VALUES(?,?)", ('total_walkies', str(val)))
        conn.commit()

def get_bookings_df(conn):
    with db_lock:
        df = pd.read_sql_query("SELECT * FROM bookings ORDER BY created_at DESC", conn)
        if not df.empty:
            # convert date columns
            for col in ['start_date','end_date','created_at']:
                df[col] = pd.to_datetime(df[col])
        return df

def add_booking(conn, client_name, phone, units, price_per_day, start_date, end_date, notes):
    with db_lock:
        booking_id = str(uuid.uuid4())[:8]
        created_at = datetime.now().isoformat()
        c = conn.cursor()
        c.execute('''
            INSERT INTO bookings(booking_id, client_name, phone, units, price_per_day, start_date, end_date, created_at, notes)
            VALUES(?,?,?,?,?,?,?,?,?)
        ''', (booking_id, client_name, phone, units, price_per_day, start_date.isoformat(), end_date.isoformat(), created_at, notes))
        conn.commit()
        return booking_id

def mark_return(conn, booking_id, returned_units, note, damaged=False):
    with db_lock:
        c = conn.cursor()
        c.execute('SELECT units, returned_units, notes FROM bookings WHERE booking_id=?', (booking_id,))
        row = c.fetchone()
        if not row:
            return False
        units, prev_returned, existing_notes = row
        new_returned = prev_returned + returned_units
        status = 'returned' if new_returned >= units else 'partial'
        
        # Build notes
        notes_list = []
        if existing_notes:
            notes_list.append(existing_notes)
        if note:
            notes_list.append(note)
        if damaged:
            notes_list.append('⚠️ DAMAGED')
        combined_note = ' | '.join(notes_list)
        
        c.execute('UPDATE bookings SET returned_units=?, status=?, notes=? WHERE booking_id=?', 
                  (new_returned, status, combined_note, booking_id))
        conn.commit()
        return True

def available_walkies(conn):
    total = get_total_walkies(conn)
    df = get_bookings_df(conn)
    if df.empty:
        return total
    
    # Calculate currently rented units (units - returned_units for non-returned statuses)
    df_active = df[df['status'] != 'returned']
    if df_active.empty:
        return total
    
    rented = (df_active['units'] - df_active['returned_units']).sum()
    return max(0, total - int(rented))

def get_rented_count(conn):
    df = get_bookings_df(conn)
    if df.empty:
        return 0
    df_active = df[df['status'] != 'returned']
    if df_active.empty:
        return 0
    return int((df_active['units'] - df_active['returned_units']).sum())

def compute_income(conn):
    df = get_bookings_df(conn)
    if df.empty:
        return 0.0
    income = 0.0
    for _, r in df.iterrows():
        units = int(r['units'])
        price = float(r['price_per_day'])
        start = r['start_date']
        end = r['end_date']
        days = max(1, (end - start).days + 1)
        income += units * price * days
    return income

def get_asset_breakdown(conn):
    """Get detailed breakdown of asset status"""
    total = get_total_walkies(conn)
    df = get_bookings_df(conn)
    
    if df.empty:
        return {
            'total': total,
            'available': total,
            'rented': 0,
            'overdue': 0,
            'maintenance': 0,
            'bookings_details': []
        }
    
    today = pd.Timestamp(datetime.now())
    df_active = df[df['status'] != 'returned'].copy()
    
    rented = 0
    overdue = 0
    damaged_count = 0
    bookings_details = []
    
    for _, r in df_active.iterrows():
        units_out = int(r['units']) - int(r['returned_units'])
        rented += units_out
        
        is_overdue = r['end_date'] < today
        is_damaged = r['notes'] and 'DAMAGED' in str(r['notes'])
        
        if is_overdue:
            overdue += units_out
        if is_damaged:
            damaged_count += 1
            
        bookings_details.append({
            'booking_id': r['booking_id'],
            'client': r['client_name'],
            'units_out': units_out,
            'end_date': r['end_date'],
            'is_overdue': is_overdue,
            'is_damaged': is_damaged,
            'status': r['status']
        })
    
    return {
        'total': total,
        'available': max(0, total - rented),
        'rented': rented,
        'overdue': overdue,
        'maintenance': damaged_count,
        'bookings_details': bookings_details
    }

# ==================== CHATBOT TOOLS ====================

def get_inventory_status(conn):
    """Get current inventory status"""
    total = get_total_walkies(conn)
    available = available_walkies(conn)
    rented = get_rented_count(conn)
    
    return {
        "total_inventory": total,
        "available_units": available,
        "rented_units": rented,
        "utilization_percentage": round((rented / total * 100), 2) if total > 0 else 0
    }

def search_bookings(conn, query=None, booking_id=None, client_name=None, status=None):
    """Search bookings by various criteria"""
    df = get_bookings_df(conn)
    
    if df.empty:
        return {"message": "No bookings found", "bookings": []}
    
    # Filter by criteria
    if booking_id:
        df = df[df['booking_id'].str.contains(booking_id, case=False, na=False)]
    if client_name:
        df = df[df['client_name'].str.contains(client_name, case=False, na=False)]
    if status:
        df = df[df['status'] == status]
    if query:
        mask = (df['client_name'].str.contains(query, case=False, na=False) | 
                df['phone'].str.contains(query, case=False, na=False) | 
                df['booking_id'].str.contains(query, case=False, na=False))
        df = df[mask]
    
    if df.empty:
        return {"message": "No matching bookings found", "bookings": []}
    
    # Convert to list of dicts
    bookings = []
    for _, r in df.iterrows():
        bookings.append({
            "booking_id": r['booking_id'],
            "client_name": r['client_name'],
            "phone": r['phone'],
            "units": int(r['units']),
            "returned_units": int(r['returned_units']),
            "price_per_day": float(r['price_per_day']),
            "start_date": r['start_date'].strftime('%Y-%m-%d'),
            "end_date": r['end_date'].strftime('%Y-%m-%d'),
            "status": r['status'],
            "notes": r['notes'] if r['notes'] else ""
        })
    
    return {"message": f"Found {len(bookings)} booking(s)", "bookings": bookings}

def get_overdue_rentals(conn):
    """Get all overdue rentals"""
    df = get_bookings_df(conn)
    
    if df.empty:
        return {"message": "No bookings found", "overdue_rentals": []}
    
    today = pd.Timestamp(datetime.now())
    df_overdue = df[(df['end_date'] < today) & (df['status'] != 'returned')]
    
    if df_overdue.empty:
        return {"message": "No overdue rentals", "overdue_rentals": []}
    
    overdue = []
    for _, r in df_overdue.iterrows():
        days_overdue = (today - r['end_date']).days
        overdue.append({
            "booking_id": r['booking_id'],
            "client_name": r['client_name'],
            "phone": r['phone'],
            "units_outstanding": int(r['units'] - r['returned_units']),
            "end_date": r['end_date'].strftime('%Y-%m-%d'),
            "days_overdue": int(days_overdue),
            "status": r['status']
        })
    
    return {"message": f"Found {len(overdue)} overdue rental(s)", "overdue_rentals": overdue}

def get_income_stats(conn):
    """Get income statistics"""
    df = get_bookings_df(conn)
    
    if df.empty:
        return {"message": "No bookings found", "total_income": 0, "breakdown": []}
    
    total_income = compute_income(conn)
    
    # Calculate income by status
    active_income = 0
    completed_income = 0
    
    for _, r in df.iterrows():
        units = int(r['units'])
        price = float(r['price_per_day'])
        start = r['start_date']
        end = r['end_date']
        days = max(1, (end - start).days + 1)
        booking_income = units * price * days
        
        if r['status'] == 'returned':
            completed_income += booking_income
        else:
            active_income += booking_income
    
    return {
        "total_projected_income": round(total_income, 2),
        "active_rentals_income": round(active_income, 2),
        "completed_rentals_income": round(completed_income, 2),
        "total_bookings": len(df),
        "active_bookings": len(df[df['status'] != 'returned'])
    }

def get_client_history(conn, client_name):
    """Get booking history for a specific client"""
    df = get_bookings_df(conn)
    
    if df.empty:
        return {"message": "No bookings found", "client_history": []}
    
    df_client = df[df['client_name'].str.contains(client_name, case=False, na=False)]
    
    if df_client.empty:
        return {"message": f"No bookings found for client: {client_name}", "client_history": []}
    
    history = []
    total_spent = 0
    
    for _, r in df_client.iterrows():
        units = int(r['units'])
        price = float(r['price_per_day'])
        start = r['start_date']
        end = r['end_date']
        days = max(1, (end - start).days + 1)
        booking_total = units * price * days
        total_spent += booking_total
        
        history.append({
            "booking_id": r['booking_id'],
            "units": units,
            "start_date": start.strftime('%Y-%m-%d'),
            "end_date": end.strftime('%Y-%m-%d'),
            "status": r['status'],
            "total_cost": round(booking_total, 2)
        })
    
    return {
        "client_name": client_name,
        "total_bookings": len(history),
        "total_spent": round(total_spent, 2),
        "booking_history": history
    }

# Define tools for OpenAI function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_inventory_status",
            "description": "Get the current inventory status including total, available, and rented walkie-talkies",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_bookings",
            "description": "Search for bookings by booking ID, client name, status, or general query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "General search query for client name, phone, or booking ID"
                    },
                    "booking_id": {
                        "type": "string",
                        "description": "Specific booking ID to search for"
                    },
                    "client_name": {
                        "type": "string",
                        "description": "Client name to search for"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["active", "partial", "returned"],
                        "description": "Booking status to filter by"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_overdue_rentals",
            "description": "Get all overdue rentals that haven't been returned yet",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_income_stats",
            "description": "Get income statistics including total, active, and completed rental income",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_client_history",
            "description": "Get booking history and total spending for a specific client",
            "parameters": {
                "type": "object",
                "properties": {
                    "client_name": {
                        "type": "string",
                        "description": "The name of the client to look up"
                    }
                },
                "required": ["client_name"]
            }
        }
    }
]

def execute_function(function_name, arguments, conn):
    """Execute the requested function with given arguments"""
    if function_name == "get_inventory_status":
        return get_inventory_status(conn)
    elif function_name == "search_bookings":
        return search_bookings(conn, **arguments)
    elif function_name == "get_overdue_rentals":
        return get_overdue_rentals(conn)
    elif function_name == "get_income_stats":
        return get_income_stats(conn)
    elif function_name == "get_client_history":
        return get_client_history(conn, **arguments)
    else:
        return {"error": "Function not found"}

def chat_with_openai(messages, conn):
    """Chat with OpenAI using function calling"""
    try:
        # Check if API key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "⚠️ OpenAI API key not set. Please set OPENAI_API_KEY environment variable."
        
        client = OpenAI(api_key=api_key,
                        base_url="https://openrouter.ai/api/v1",)
        
        # Initial API call
        response = client.chat.completions.create(
            model="minimax/minimax-m2:free",  # or "gpt-4" for better results
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        
        # Check if the model wants to call a function
        if response_message.tool_calls:
            # Add the assistant's message to conversation
            messages.append(response_message)
            
            # Execute each function call
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Execute the function
                function_response = execute_function(function_name, function_args, conn)
                
                # Add function response to messages
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response)
                })
            
            # Get final response from model
            second_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            
            return second_response.choices[0].message.content
        else:
            return response_message.content
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ----------------- Streamlit UI -----------------

st.set_page_config(page_title="Walkie Rental Manager", layout='wide')
conn = get_conn()
init_db(conn)

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "system", "content": """You are a helpful assistant for a walkie-talkie rental management system. 
You have access to database tools to help answer questions about inventory, bookings, income, and clients.
Be friendly, concise, and professional. Use emojis sparingly. When showing data, format it clearly."""}
    ]
if 'show_chatbot' not in st.session_state:
    st.session_state.show_chatbot = False

st.title("📻 Walkie Rental Manager")

# Chatbot toggle button
col_title1, col_title2 = st.columns([6,1])
with col_title2:
    if st.button("💬 AI Chat" if not st.session_state.show_chatbot else "❌ Close"):
        st.session_state.show_chatbot = not st.session_state.show_chatbot

# Chatbot interface (shown as overlay)
if st.session_state.show_chatbot:
    with st.container():
        st.markdown("---")
        st.subheader("🤖 AI Assistant (Powered by OpenAI)")
        st.caption("Ask me anything about your inventory, bookings, income, or clients!")
        
        # Check if API key is set
        if not os.getenv("OPENAI_API_KEY"):
            st.error("⚠️ OPENAI_API_KEY environment variable not set. Please set it to use the chatbot.")
            st.code("export OPENAI_API_KEY='your-api-key-here'", language="bash")
        
        # Chat display
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.markdown(f"**AI:** {msg['content']}")
        
        # Chat input
        col1, col2 = st.columns([5,1])
        with col1:
            user_input = st.text_input("Ask me anything...", key="chat_input", 
                                      placeholder="e.g., How many walkies are available? or Show me overdue rentals")
        with col2:
            send_btn = st.button("Send", use_container_width=True)
        
        if send_btn and user_input:
            # Add user message
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input
            })
            st.session_state.chat_messages.append({
                'role': 'user',
                'content': user_input
            })
            
            # Get bot response with spinner
            with st.spinner("🤔 Thinking..."):
                response = chat_with_openai(st.session_state.chat_messages, conn)
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response
                })
                st.session_state.chat_messages.append({
                    'role': 'assistant',
                    'content': response
                })
            
            st.rerun()
        
        col_clear1, col_clear2 = st.columns([1,1])
        with col_clear1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.chat_messages = [st.session_state.chat_messages[0]]  # Keep system message
                st.rerun()
        
        with col_clear2:
            if st.button("Sample Questions", use_container_width=True):
                st.info("""
**Try asking:**
- How many walkies are available?
- Show me all overdue rentals
- What's our total income?
- Find bookings for John
- Search for booking abc123
- Show me active rentals
- Get client history for Sarah
                """)
        
        st.markdown("---")

# Sidebar: Settings & Add Booking
with st.sidebar:
    st.header("Settings")
    total_walkies = get_total_walkies(conn)
    new_total = st.number_input("Total walkies in inventory", min_value=1, value=total_walkies, key='total_setting')
    if new_total != total_walkies:
        set_total_walkies(conn, int(new_total))
        st.success(f"Total inventory updated to {new_total}")
        st.rerun()

    st.markdown("---")
    st.header("Quick Add Booking")
    with st.form("quick_add"):
        client_name = st.text_input("Client name")
        phone = st.text_input("Phone")
        available = available_walkies(conn)
        units = st.number_input("Units", min_value=1, max_value=max(1, available), value=1)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", value=date.today())
        with col2:
            end_date = st.date_input("End date", value=date.today())
        price_per_day = st.number_input("Price per unit per day", min_value=0.0, value=10.0, step=1.0)
        notes = st.text_area("Notes (optional)")
        submitted = st.form_submit_button("Add booking")
        
        if submitted:
            # Validation
            if not client_name or not phone:
                st.error("Client name and phone are required!")
            elif end_date < start_date:
                st.error("End date must be after start date!")
            elif units > available_walkies(conn):
                st.error(f"Not enough available walkies. Available: {available_walkies(conn)}")
            else:
                bid = add_booking(conn, client_name, phone, units, price_per_day, start_date, end_date, notes)
                st.success(f"✅ Booking added (ID: {bid})")
                st.rerun()

# Main area - Asset Status Component
st.header("📊 Current Asset Status")
asset_data = get_asset_breakdown(conn)

# Visual asset breakdown
col1, col2, col3, col4 = st.columns(4)
col1.metric("🏢 Total Inventory", asset_data['total'])
col2.metric("✅ Available", asset_data['available'], 
            delta=None if asset_data['available'] > 0 else "⚠️ None available")
col3.metric("📤 Currently Rented", asset_data['rented'])
col4.metric("⏰ Overdue Units", asset_data['overdue'],
            delta="Action needed" if asset_data['overdue'] > 0 else None)

# Asset utilization bar
if asset_data['total'] > 0:
    utilization = (asset_data['rented'] / asset_data['total']) * 100
    st.progress(utilization / 100)
    st.caption(f"Utilization: {utilization:.1f}% ({asset_data['rented']} of {asset_data['total']} units in use)")

# Detailed asset breakdown
if asset_data['bookings_details']:
    st.markdown("### 📋 Active Rentals Breakdown")
    breakdown_df = pd.DataFrame(asset_data['bookings_details'])
    breakdown_df['end_date'] = pd.to_datetime(breakdown_df['end_date']).dt.strftime('%Y-%m-%d')
    breakdown_df['Days Left'] = (pd.to_datetime(breakdown_df['end_date']) - pd.Timestamp(datetime.now())).dt.days
    
    display_breakdown = breakdown_df[['booking_id', 'client', 'units_out', 'end_date', 'Days Left', 'status']]
    display_breakdown.columns = ['Booking ID', 'Client', 'Units Out', 'End Date', 'Days Left', 'Status']
    
    # Color code based on status
    def highlight_overdue(row):
        if row['Days Left'] < 0:
            return ['background-color: #ffcccc'] * len(row)
        elif row['Days Left'] <= 3:
            return ['background-color: #fff3cd'] * len(row)
        return [''] * len(row)
    
    st.dataframe(display_breakdown.style.apply(highlight_overdue, axis=1), use_container_width=True)

st.markdown("---")

# Dashboard section
colA, colB = st.columns([2,1])

with colA:
    st.header("Dashboard Metrics")
    total = get_total_walkies(conn)
    df = get_bookings_df(conn)
    rented_units = get_rented_count(conn)
    available = available_walkies(conn)
    income = compute_income(conn)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Walkies", total)
    c2.metric("Rented Out", rented_units)
    c3.metric("Available", available)
    c4.metric("Projected Income", f"${income:.2f}")

    st.markdown("---")
    st.subheader("All Bookings")
    if df.empty:
        st.info("No bookings yet. Add one from the sidebar!")
    else:
        today = pd.Timestamp(datetime.now())
        df['due_in_days'] = (df['end_date'] - today).dt.days
        display = df[['booking_id','client_name','phone','units','start_date','end_date','returned_units','status','price_per_day','notes','due_in_days']].copy()
        display['start_date'] = display['start_date'].dt.strftime('%Y-%m-%d')
        display['end_date'] = display['end_date'].dt.strftime('%Y-%m-%d')
        st.dataframe(display, use_container_width=True, height=300)

    st.markdown("---")
    st.subheader("Returns / Update Booking")
    with st.form("returns_form"):
        bookings = get_bookings_df(conn)
        active_bookings = bookings[bookings['status'] != 'returned']
        
        if active_bookings.empty:
            st.write("No active bookings to update.")
            st.form_submit_button("No Action Available", disabled=True)
        else:
            active_options = active_bookings['booking_id'].tolist()
            sel = st.selectbox("Select booking", active_options)
            sel_row = active_bookings[active_bookings['booking_id'] == sel].iloc[0]
            
            remaining = int(sel_row['units'] - sel_row['returned_units'])
            st.write(f"**Client:** {sel_row['client_name']} | **Units rented:** {sel_row['units']} | **Already returned:** {sel_row['returned_units']} | **Still out:** {remaining}")
            
            ret_units = st.number_input("Units to return now", min_value=1, max_value=remaining, value=min(1, remaining))
            damaged = st.checkbox("Mark as damaged")
            return_note = st.text_area("Return notes (optional)")