# Use the Pixi image as the base
FROM ghcr.io/prefix-dev/pixi:latest

# Install curl
RUN apt-get update && apt-get install -y curl

# Set the working directory
WORKDIR /app

# Copy your project files
COPY . .

# Install dependencies using Pixi
RUN pixi install

# Expose the Streamlit port
EXPOSE 8501

# Start Streamlit
CMD ["pixi run start"]