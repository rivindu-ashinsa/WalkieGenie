# Walkie-Talkie Rental Management System

A comprehensive web-based application for managing walkie-talkie rentals, featuring real-time inventory tracking, client management, analytics, and an AI-powered assistant.

## Key Features

- **Real-time Inventory Management**: Track total, rented, and available walkie units with automatic availability calculations
- **Booking Lifecycle Management**: Complete rental workflow from booking creation to return processing
- **Client History**: Comprehensive rental history and client relationship management
- **Analytics Dashboard**: Income tracking, overdue rental alerts, and inventory status reports
- **AI-Powered Assistant**: OpenAI-integrated chatbot for system queries and assistance
- **Database Analytics**: Advanced filtering and search capabilities across all booking data
- **Atomic Transactions**: Ensures data consistency during rental operations and returns

## Tech Stack

- **Backend**: Python 3.x
- **Database**: SQLite3
- **Web Framework**: Streamlit
- **Data Processing**: pandas
- **AI Integration**: OpenAI API
- **Configuration**: Environment variables (.env)

## Project Structure

```
walkie-rental-system/
├── .gitignore                 # Git ignore configuration
├── main.py                    # Core business logic and data layer
├── walkie_rental.db           # SQLite database file
├── walkie_rental_app.py       # Streamlit UI application
└── .env                       # Environment variables (OpenAI API key)
```

### Core Components

- **Database Layer**: SQLite operations with atomic transactions
- **Business Logic**: Rental lifecycle, inventory control, and analytics
- **API Integration**: OpenAI chatbot functionality
- **UI Layer**: Streamlit web interface

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for chatbot functionality)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd walkie-rental-system
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit pandas openai python-dotenv
   ```

3. **Configure environment variables**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Initialize the database**
   The SQLite database will be created automatically on first run with:
   - `bookings` table for rental records
   - `settings` table for system configuration
   - Default inventory: 50 walkie units

## Usage

### Running the Application

Start the Streamlit web interface:

```bash
streamlit run walkie_rental_app.py
```

Access the application at `http://localhost:8501`.

### Key Operations

- **Add New Booking**: Create rental records with customer details, rental period, and pricing
- **Process Returns**: Mark units as returned with optional damage notes
- **View Analytics**: Access income reports, overdue rentals, and inventory status
- **Client Management**: Search and review client rental history
- **AI Assistant**: Use the integrated chatbot for system queries and assistance

### Database Management

The system automatically manages:
- Booking status (active, returned, overdue)
- Inventory availability calculations
- Revenue computation and time-range analytics
- Overdue rental identification

### API Integration

```python
# Example: Using the core functions
from main import add_booking, get_bookings_df, mark_return

# Add a new booking
booking_id = add_booking(
    client_name="John Doe",
    phone="555-0123",
    units=2,
    price_per_day=15.00,
    start_date="2024-01-15",
    end_date="2024-01-20"
)

# Process a return
mark_return(
    booking_id=booking_id,
    returned_units=2,
    note="Returned in good condition"
)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure all database operations maintain data integrity and include appropriate error handling.

## License

This project is licensed under the MIT License - see the project repository for details.