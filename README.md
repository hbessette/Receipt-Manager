Currently, Receipt Manager is capable of preprocessing receipt images using techniques of blurring, rotating, cropping, inverting, resizing, gamma correcting, and noise removing to provide PyTesseract a clear image. PyTesseract then reads this image and outputs the raw text from a receipt.

Development Plan:
  - Identifying key fields to extract such as store name, data and time, list of purchasd items (quantity, name, price), total amount, tax amount, payment method.
  - Using pattern matching, parse data based on keywords or structure and handle errors such as fields not found.
  - Using SQLite, store data with a database schema that mirrors the fields listed above.
  - Implement a function that queries the database based on user input and exports the data to a .csv file.
  - Test the full pipeline with a variety of receipt images.
  - Optimize potential inefficiencies within the pipeline.
  - Implement a simple user interface for ease of use.
