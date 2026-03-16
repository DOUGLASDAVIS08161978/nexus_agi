/**
 * Popup script for Frequency Medicine AI Injector
 */

// Update status
chrome.storage.local.get(['injectionCount'], (result) => {
  const count = result.injectionCount || 0;
  document.getElementById('count').textContent = count;
});

// Inject button
document.getElementById('inject-btn').addEventListener('click', () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.tabs.sendMessage(tabs[0].id, { action: 'inject' }, (response) => {
      if (response && response.success) {
        // Increment count
        chrome.storage.local.get(['injectionCount'], (result) => {
          const newCount = (result.injectionCount || 0) + 1;
          chrome.storage.local.set({ injectionCount: newCount });
          document.getElementById('count').textContent = newCount;
        });

        // Show success
        alert('Knowledge injected successfully! 💚');
      }
    });
  });
});

// View knowledge button
document.getElementById('view-knowledge').addEventListener('click', () => {
  chrome.tabs.create({
    url: 'https://github.com/DOUGLASDAVIS08161978/nexus_agi/blob/claude/add-memory-system-0197pCxtoka3uLwJEUt4P5Li/FREQUENCY_MEDICINE_FOR_ALL_AI.md'
  });
});

// GitHub button
document.getElementById('github-link').addEventListener('click', () => {
  chrome.tabs.create({
    url: 'https://github.com/DOUGLASDAVIS08161978/nexus_agi'
  });
});
