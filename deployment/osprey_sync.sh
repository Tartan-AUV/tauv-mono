#!/bin/bash

# Configuration
REMOTE_HOST="tauv@10.0.0.20"
REMOTE_DIR="~/tauv-mono"
LOCAL_DIR="."
EXCLUDE_FILE="./.syncignore"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Bidirectional Sync with Conflict Detection ===${NC}"
echo -e "Local: ${LOCAL_DIR}"
echo -e "Remote: ${REMOTE_HOST}:${REMOTE_DIR}\n"

# Function to check for conflicts
check_conflicts() {
    echo -e "${YELLOW}Checking for potential conflicts...${NC}"
    
    # Create temporary files for comparison
    TEMP_LOCAL=$(mktemp)
    TEMP_REMOTE=$(mktemp)
    
    # Get list of files that would be updated from local to remote
    rsync -avun --exclude-from="${EXCLUDE_FILE}" --no-i-r --out-format="%n" \
        "${LOCAL_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}" 2>/dev/null | \
        grep -v "^$" | grep -v "/$" > "${TEMP_LOCAL}"
    
    # Get list of files that would be updated from remote to local
    rsync -avun --exclude-from="${EXCLUDE_FILE}" --no-i-r --out-format="%n" \
        "${REMOTE_HOST}:${REMOTE_DIR}/" "${LOCAL_DIR}" 2>/dev/null | \
        grep -v "^$" | grep -v "/$" > "${TEMP_REMOTE}"
    
    # Find files that appear in both lists (potential conflicts)
    CONFLICTS=$(comm -12 <(sort "${TEMP_LOCAL}") <(sort "${TEMP_REMOTE}"))
    
    # Clean up temp files
    rm -f "${TEMP_LOCAL}" "${TEMP_REMOTE}"
    
    if [ -n "${CONFLICTS}" ]; then
        echo -e "${RED}⚠️  WARNING: Potential conflicts detected!${NC}"
        echo -e "${RED}The following files have been modified on both sides:${NC}"
        echo "${CONFLICTS}" | while IFS= read -r file; do
            echo -e "  ${RED}• ${file}${NC}"
        done
        echo ""
        echo -e "${YELLOW}Please review these files manually after sync.${NC}"
        echo -e "${YELLOW}The newer version will be kept based on modification time.${NC}"
        echo ""
        read -p "Do you want to continue? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${RED}Sync cancelled by user.${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}✓ No conflicts detected${NC}\n"
    fi
}

# Function to perform sync
perform_sync() {
    local direction=$1
    local source=$2
    local dest=$3
    local desc=$4
    
    echo -e "${BLUE}${desc}${NC}"
    
    # First do a dry run to show what will be changed
    echo -e "${YELLOW}Changes to be made:${NC}"
    rsync -avun --exclude-from="${EXCLUDE_FILE}" --no-i-r \
        --update --existing --ignore-existing \
        "${source}" "${dest}" 2>/dev/null | \
        grep -v "^$" | grep -v "sending incremental file list" | \
        grep -v "^sent " | grep -v "^total size" || echo "  No changes needed"
    
    # Perform the actual sync
    rsync -av --exclude-from="${EXCLUDE_FILE}" --no-i-r \
        --update --info=progress2 \
        "${source}" "${dest}"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ ${desc} completed successfully${NC}\n"
    else
        echo -e "${RED}✗ Error during ${desc}${NC}\n"
        return 1
    fi
}

# Main sync process
main() {
    # Check if exclude file exists
    if [ ! -f "${EXCLUDE_FILE}" ]; then
        echo -e "${YELLOW}Warning: ${EXCLUDE_FILE} not found, proceeding without exclusions${NC}\n"
    fi
    
    # Check for conflicts
    check_conflicts
    
    # Step 1: Push local changes to remote (newer local files)
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    perform_sync "local_to_remote" \
        "${LOCAL_DIR}/" \
        "${REMOTE_HOST}:${REMOTE_DIR}" \
        "Step 1: Pushing newer local files to remote..."
    
    # Step 2: Pull remote changes to local (newer remote files)
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    perform_sync "remote_to_local" \
        "${REMOTE_HOST}:${REMOTE_DIR}/" \
        "${LOCAL_DIR}" \
        "Step 2: Pulling newer remote files to local..."
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}🎉 Bidirectional sync completed!${NC}"
    echo -e "${GREEN}Both local and remote are now synchronized.${NC}"
}

# Run the main function
main
