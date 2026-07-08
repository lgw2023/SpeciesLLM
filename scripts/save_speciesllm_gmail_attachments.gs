const QUERY = 'from:17621223203@163.com has:attachment';
const FOLDER_NAME = 'SpeciesLLM Gmail attachments';
const BATCH_SIZE = 50;
const OFFSET_KEY = 'speciesllm_offset';

function shouldSave_(filename) {
  const name = String(filename || '').toLowerCase().trim();

  return (
    name.startsWith('data') ||
    name.endsWith('.json') ||
    name.endsWith('json') ||
    name.endsWith('.jsonpart') ||
    name.includes('.json')
  );
}

function getTargetFolder_() {
  const root = DriveApp.getRootFolder();
  const folders = root.getFoldersByName(FOLDER_NAME);
  return folders.hasNext() ? folders.next() : root.createFolder(FOLDER_NAME);
}

function getExistingNames_(folder) {
  const names = new Set();
  const files = folder.getFiles();

  while (files.hasNext()) {
    names.add(files.next().getName());
  }

  return names;
}

function saveAttachmentsToDrive() {
  const folder = getTargetFolder_();
  const existingNames = getExistingNames_(folder);
  const props = PropertiesService.getScriptProperties();

  let offset = Number(props.getProperty(OFFSET_KEY) || '0');

  let totalThreads = 0;
  let totalSeenAttachments = 0;
  let totalMatched = 0;
  let totalSaved = 0;
  let totalSkippedDuplicate = 0;
  let totalIgnored = 0;

  const startedAt = Date.now();
  const MAX_RUNTIME_MS = 5 * 60 * 1000; // 主动留安全余量，避免运行超时

  while (Date.now() - startedAt < MAX_RUNTIME_MS) {
    const threads = GmailApp.search(QUERY, offset, BATCH_SIZE);

    if (threads.length === 0) {
      props.deleteProperty(OFFSET_KEY);
      console.log('完成：没有更多匹配线程。');
      break;
    }

    for (const thread of threads) {
      for (const message of thread.getMessages()) {
        const attachments = message.getAttachments({
          includeInlineImages: false,
          includeAttachments: true,
        });

        for (const attachment of attachments) {
          totalSeenAttachments++;

          const filename = attachment.getName();

          if (!shouldSave_(filename)) {
            totalIgnored++;
            continue;
          }

          totalMatched++;

          if (existingNames.has(filename)) {
            totalSkippedDuplicate++;
            continue;
          }

          const blob = attachment.copyBlob().setName(filename);
          folder.createFile(blob);
          existingNames.add(filename);
          totalSaved++;
        }
      }
    }

    totalThreads += threads.length;
    offset += threads.length;

    console.log(`已处理到 offset=${offset}`);

    if (threads.length < BATCH_SIZE) {
      props.deleteProperty(OFFSET_KEY);
      console.log('完成：已处理最后一批。');
      break;
    }

    props.setProperty(OFFSET_KEY, String(offset));
  }

  console.log(`本次处理线程=${totalThreads}`);
  console.log(`看到附件=${totalSeenAttachments}`);
  console.log(`命中文件名规则=${totalMatched}`);
  console.log(`新保存=${totalSaved}`);
  console.log(`跳过重复=${totalSkippedDuplicate}`);
  console.log(`忽略=${totalIgnored}`);
  console.log(`当前 offset=${props.getProperty(OFFSET_KEY) || '已清空，表示完成'}`);
}

function resetProgress() {
  PropertiesService.getScriptProperties().deleteProperty(OFFSET_KEY);
  console.log('已重置进度。现在重新运行 saveAttachmentsToDrive。');
}