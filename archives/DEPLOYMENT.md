# Deployment Guide for RAG Application

Deploy the Celebrity Cruises RAG Assistant to **rag.guillaume.genois.ca**

## Prerequisites

- Ubuntu server with SSH access
- Domain `rag.guillaume.genois.ca` pointing to your server IP
- Root or sudo access

## Step 1: Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3 python3-pip python3-venv nginx certbot python3-certbot-nginx git

# Create application directory
sudo mkdir -p /var/www/rag.guillaume.genois.ca
sudo chown $USER:$USER /var/www/rag.guillaume.genois.ca
cd /var/www/rag.guillaume.genois.ca
```

## Step 2: Clone Repository

```bash
# Clone your repository
git clone https://github.com/Guigui031/AIAssistantWithRAG.git .

# Or upload files manually
# rsync -avz --exclude 'chroma*' /path/to/local/repo/ user@server:/var/www/rag.guillaume.genois.ca/
```

## Step 3: Python Environment Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 4: Environment Configuration

```bash
# Create .env file
nano .env
```

Add your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Step 5: Create Documents Directory

```bash
# Create documents directory
mkdir -p documents

# Upload your CSV files
# You can upload via SCP:
# scp celebrity-cruises.csv user@server:/var/www/rag.guillaume.genois.ca/documents/
```

## Step 6: Test Application Locally

```bash
# Test the app
streamlit run app.py

# If it works, you'll see:
# You can now view your Streamlit app in your browser.
# Local URL: http://localhost:8501

# Press Ctrl+C to stop
```

## Step 7: Configure Systemd Service

```bash
# Copy service file
sudo cp systemd/rag-app.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start service
sudo systemctl enable rag-app
sudo systemctl start rag-app

# Check status
sudo systemctl status rag-app
```

## Step 8: Configure Nginx

```bash
# Copy nginx configuration
sudo cp nginx/rag.conf /etc/nginx/sites-available/rag.guillaume.genois.ca

# Create symbolic link
sudo ln -s /etc/nginx/sites-available/rag.guillaume.genois.ca /etc/nginx/sites-enabled/

# Test nginx configuration
sudo nginx -t

# If test passes, reload nginx
sudo systemctl reload nginx
```

## Step 9: Setup SSL with Let's Encrypt

```bash
# Obtain SSL certificate
sudo certbot --nginx -d rag.guillaume.genois.ca

# Follow the prompts and agree to terms
# Certbot will automatically configure SSL in nginx

# Test auto-renewal
sudo certbot renew --dry-run
```

## Step 10: Verify Deployment

1. Open browser and visit: https://rag.guillaume.genois.ca
2. You should see the RAG application interface
3. Click "Initialize/Reload RAG System" in the sidebar
4. Try a query like "Show me cruises in October 2026"

## Useful Commands

### View logs
```bash
# Application logs
sudo journalctl -u rag-app -f

# Nginx logs
sudo tail -f /var/log/nginx/rag_access.log
sudo tail -f /var/log/nginx/rag_error.log
```

### Restart services
```bash
# Restart application
sudo systemctl restart rag-app

# Restart nginx
sudo systemctl restart nginx
```

### Update application
```bash
cd /var/www/rag.guillaume.genois.ca
git pull
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart rag-app
```

### Add new documents
```bash
# Upload CSV file to documents directory
scp new-data.csv user@server:/var/www/rag.guillaume.genois.ca/documents/

# Or use the web interface upload feature
```

### Rebuild vector database
```bash
# SSH into server
cd /var/www/rag.guillaume.genois.ca

# Remove old database
rm -rf chroma_structured_db

# Restart service (will rebuild on next query)
sudo systemctl restart rag-app
```

## Firewall Configuration

```bash
# Allow SSH, HTTP, and HTTPS
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

## Monitoring

### Check if app is running
```bash
# Check systemd service
sudo systemctl status rag-app

# Check if port 8501 is listening
sudo netstat -tulpn | grep 8501

# Check nginx status
sudo systemctl status nginx
```

### Performance monitoring
```bash
# CPU and memory usage
htop

# Disk usage
df -h
```

## Troubleshooting

### App won't start
```bash
# Check logs
sudo journalctl -u rag-app -n 50

# Check if port is already in use
sudo lsof -i :8501

# Manually test
cd /var/www/rag.guillaume.genois.ca
source venv/bin/activate
streamlit run app.py
```

### 502 Bad Gateway
- Check if streamlit app is running: `sudo systemctl status rag-app`
- Check nginx logs: `sudo tail -f /var/log/nginx/rag_error.log`

### SSL certificate issues
```bash
# Renew certificate manually
sudo certbot renew

# Check certificate status
sudo certbot certificates
```

### Out of memory
```bash
# Check memory usage
free -h

# Add swap if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Security Best Practices

1. **Keep .env secure**: Never commit `.env` to Git
2. **Regular updates**: `sudo apt update && sudo apt upgrade`
3. **Firewall**: Only allow necessary ports
4. **SSH keys**: Use SSH keys instead of passwords
5. **Backups**: Regular backup of `/var/www/rag.guillaume.genois.ca/documents/`

## Backup Strategy

```bash
# Backup documents
tar -czf backup-$(date +%Y%m%d).tar.gz documents/

# Download backup
scp user@server:/var/www/rag.guillaume.genois.ca/backup-*.tar.gz ./
```

## Quick Deployment Script

Save this as `deploy.sh`:

```bash
#!/bin/bash
set -e

echo "🚀 Deploying RAG Application..."

# Pull latest code
git pull

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
pip install -r requirements.txt

# Restart service
sudo systemctl restart rag-app

echo "✅ Deployment complete!"
echo "Visit: https://rag.guillaume.genois.ca"
```

Make executable: `chmod +x deploy.sh`

Run: `./deploy.sh`