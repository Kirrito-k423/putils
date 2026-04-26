---
name: "pjlab-web-terminal"
description: "Operate PJLab pod web terminal via Chrome DevTools MCP: connect, authenticate, simulate keyboard input via CDP, redirect output to mounted dirs, read from IDE. Invoke when user needs to run commands on PJLab remote pod."
---

# PJLab Web Terminal Skill

通过 Chrome DevTools MCP 连接仪电智算云（PJLab）Pod Web 终端，使用 CDP 模拟键盘输入执行命令，将输出重定向到挂载目录，在本地 IDE 中直接读取结果。

## 核心工作流

```
连接终端 → 认证登录 → CDP模拟输入命令 → 输出重定向到挂载目录 → IDE读取文件
```

**为什么需要这个 Skill？** PJLab 的终端是 xterm.js + WebGL Canvas 渲染的 Web 终端，无法通过 DOM 直接读取终端文本输出。最可靠的方式是将命令输出重定向到 Pod 上被挂载到本地的目录，然后在 IDE 中直接读取该文件。

---

## 1. 前置条件

### 1.1 Chrome DevTools MCP

必须在 `~/Library/Application Support/Trae CN/User/mcp.json` 中配置：

```json
{
  "Chrome DevTools MCP": {
    "command": "npx",
    "args": ["-y", "chrome-devtools-mcp@latest"],
    "env": {},
    "fromGalleryId": "byted-mcp.chrome-devtools-mcp"
  }
}
```

### 1.2 网络要求

**必须关闭系统代理**。WebSocket 升级请求经过代理（如 Clash `127.0.0.1:7890`）时会返回 HTTP 200 而非 101 Switching Protocols，导致连接失败。

解决方案（二选一）：
- 关闭全局代理后再使用
- 在 MCP 配置中添加 `"--chromeArg=--no-proxy-server"` 到 args 数组（但这会禁用所有站点的代理）

### 1.3 挂载目录

Pod 的存储目录需要挂载到本地，这样重定向到该目录的文件才能在 IDE 中读取。通常 PJLab 的 Pod 存储会自动挂载到本地开发环境中。

---

## 2. 连接终端

### 2.1 导航到终端页面

终端 URL 格式：

```
https://console.d.pjlab.org.cn/ecp/terminal?cluster=<CLUSTER_ID>&containerName=<CONTAINER>&namespace=<NAMESPACE>&podName=<POD_NAME>
```

使用 Chrome DevTools MCP 的 `evaluate_script` 导航：

```javascript
window.location.href = 'https://console.d.pjlab.org.cn/ecp/terminal?cluster=019bffe7-fb5a-7c30-bb75-4abe8621192e&containerName=worker&namespace=default&podName=tsj-lm-kl-1-worker-0';
```

等待 8-10 秒让页面加载完成。

### 2.2 验证页面加载

```javascript
{ url: window.location.href, title: document.title }
// 期望 title: "仪电智算云-控制台"
```

如果 URL 被重定向到 `signin.d.pjlab.org.cn`，说明需要登录，参见第 3 节。

### 2.3 验证终端 WebSocket 连接

等待 10-15 秒后，使用 `list_console_messages`（设置 `includePreservedMessages=true`）检查：

| 控制台消息 | 含义 |
|-----------|------|
| `render xterm use webgl` | xterm 已用 WebGL 渲染器初始化 |
| `connect socket success` | WebSocket 连接成功 ✅ |
| `heartbeat start` | 心跳机制已启动 ✅ |
| `get token failed` | 认证 token 缺失，终端无法连接 ❌ |
| `Unexpected response code: 200` | 代理问题导致 WebSocket 升级失败 ❌ |

### 2.4 验证终端 DOM 元素

```javascript
const termDiv = document.querySelector('.terminal.xterm');
const textarea = document.querySelector('.xterm-helper-textarea');
// 两者都应存在
```

---

## 3. 认证登录

### 3.1 IAM 用户登录流程

如果页面重定向到 `signin.d.pjlab.org.cn`：

1. 点击 **"IAM用户登录"** 标签页
2. 填写企业名称（如 `ailabdev`）
3. 填写用户名（如 `linyifei.p`）
4. 填写密码
5. 点击登录按钮

### 3.2 使用 CDP 操作登录表单

```javascript
// 切换到 IAM 登录标签
const iamTab = document.querySelector('.iam-login-tab');
if (iamTab) iamTab.click();

// 填写表单 - 使用 Chrome DevTools MCP 的 fill 工具
// 1. 获取页面快照找到输入框的 uid
// 2. 使用 fill 工具填入企业名称、用户名、密码
// 3. 点击登录按钮
```

### 3.3 登录后

登录成功后会自动跳转回终端页面。等待 10-15 秒，按 2.3 节验证 WebSocket 连接状态。

---

## 4. CDP 模拟命令输入

xterm.js 通过 `.xterm-helper-textarea` 接收键盘输入。**关键：文本输入用 `document.execCommand('insertText')`，特殊按键用 `KeyboardEvent`。**

### 4.1 输入文本

```javascript
const textarea = document.querySelector('.xterm-helper-textarea');
textarea.focus();

const command = 'ls /home > /tmp/output.txt 2>&1';
for (const char of command) {
    document.execCommand('insertText', false, char);
    await new Promise(r => setTimeout(r, 50));
}
```

### 4.2 按 Enter 执行命令

```javascript
const textarea = document.querySelector('.xterm-helper-textarea');
textarea.focus();
textarea.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Enter', code: 'Enter', keyCode: 13, which: 13,
    charCode: 0, bubbles: true, cancelable: true, composed: true
}));
```

### 4.3 特殊按键

| 按键 | 代码 |
|------|------|
| Ctrl+C | `new KeyboardEvent('keydown', {key:'c', code:'KeyC', keyCode:67, which:67, ctrlKey:true, bubbles:true, cancelable:true})` |
| Ctrl+L (清屏) | `new KeyboardEvent('keydown', {key:'l', code:'KeyL', keyCode:76, which:76, ctrlKey:true, bubbles:true, cancelable:true})` |
| Tab (补全) | `new KeyboardEvent('keydown', {key:'Tab', code:'Tab', keyCode:9, which:9, bubbles:true, cancelable:true})` |
| Backspace | `new KeyboardEvent('keydown', {key:'Backspace', code:'Backspace', keyCode:8, which:8, bubbles:true, cancelable:true})` |
| PageDown | `new KeyboardEvent('keydown', {key:'PageDown', code:'PageDown', keyCode:34, which:34, bubbles:true, cancelable:true})` |
| ArrowUp (历史) | `new KeyboardEvent('keydown', {key:'ArrowUp', code:'ArrowUp', keyCode:38, which:38, bubbles:true, cancelable:true})` |

### 4.4 封装的命令执行函数

```javascript
async function runCommand(cmd) {
    const textarea = document.querySelector('.xterm-helper-textarea');
    textarea.focus();
    for (const char of cmd) {
        document.execCommand('insertText', false, char);
        await new Promise(r => setTimeout(r, 30));
    }
    await new Promise(r => setTimeout(r, 100));
    textarea.dispatchEvent(new KeyboardEvent('keydown', {
        key: 'Enter', code: 'Enter', keyCode: 13, which: 13,
        charCode: 0, bubbles: true, cancelable: true, composed: true
    }));
}
```

---

## 5. 输出重定向到挂载目录（核心方法）

### 5.1 原理

Pod 的存储目录通常会被挂载到本地开发环境。将命令输出重定向到挂载目录中的文件，就可以在 IDE 中直接读取该文件，**完全绕过 WebGL Canvas 无法读取 DOM 文本的问题**。

### 5.2 执行命令并重定向输出

```javascript
// 在终端中执行命令，将输出重定向到挂载目录的文件
await runCommand('ls /home > /mnt/pjlab/terminal_output.txt 2>&1');
// 等待命令执行完成
await new Promise(r => setTimeout(r, 3000));
```

**常用重定向模式：**

```bash
# 标准输出重定向
command > /mnt/pjlab/terminal_output.txt

# 标准输出+错误都重定向
command > /mnt/pjlab/terminal_output.txt 2>&1

# 追加模式
command >> /mnt/pjlab/terminal_output.txt 2>&1

# 多条命令输出到同一文件
{ command1; command2; } > /mnt/pjlab/terminal_output.txt 2>&1
```

### 5.3 确定挂载目录

挂载目录路径取决于 Pod 配置。常见路径：
- `/mnt/` 下的子目录
- Pod 的工作目录下的子目录
- 通过 `df -h` 或 `mount` 命令查看挂载点

可以先在终端执行 `df -h > /tmp/mounts.txt 2>&1`，然后通过 Canvas 截图方式读取（参见第 6 节），确定挂载目录。

### 5.4 在 IDE 中读取输出文件

命令执行完成后，在 IDE 中直接读取重定向的文件：

```
# 文件路径取决于挂载目录映射
# 例如如果 /mnt/pjlab 映射到本地的 /Users/Zhuanz/work/data/pjlab/
# 则读取 /Users/Zhuanz/work/data/pjlab/terminal_output.txt
```

### 5.5 完整示例：查看 /home 下用户

```
步骤 1: 连接终端（第 2 节）
步骤 2: 验证 WebSocket 连接（第 2.3 节）
步骤 3: 执行命令
        await runCommand('ls /home > /mnt/pjlab/terminal_output.txt 2>&1');
步骤 4: 等待 3 秒
步骤 5: 在 IDE 中读取 /mnt/pjlab/terminal_output.txt 对应的本地文件
步骤 6: 报告结果
```

---

## 6. 备用方法：Canvas 截图 + OCR

当无法使用文件重定向时（如挂载目录不可用），可通过 Canvas 截图 + OCR 读取终端输出。

### 6.1 截图

```javascript
// 等待命令输出渲染完成
await new Promise(r => setTimeout(r, 3000));

// 捕获终端 Canvas（index 1 是主渲染 Canvas，index 0 是链接层）
const canvas = document.querySelectorAll('canvas')[1];
const dataUrl = canvas.toDataURL('image/png');
```

### 6.2 保存截图到本地

```javascript
canvas.toBlob((blob) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'terminal_output.png';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}, 'image/png');
```

### 6.3 OCR 识别

需要安装依赖：

```bash
pip install Pillow pytesseract
# macOS: brew install tesseract
# Ubuntu: apt install tesseract-ocr
```

```python
from PIL import Image, ImageOps
import pytesseract

img = Image.open('terminal_output.png')
# 深色主题终端需要反转颜色
img = ImageOps.invert(img.convert('RGB'))
text = pytesseract.image_to_string(img, config='--psm 6')
print(text)
```

### 6.4 长输出滚动截取

```javascript
// 向下滚动终端
textarea.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'PageDown', code: 'PageDown', keyCode: 34, which: 34,
    bubbles: true, cancelable: true
}));
// 等待渲染后再次截图
```

---

## 7. 故障排除

### WebSocket 连接失败（"Unexpected response code: 200"）

**原因：** 系统代理拦截了 WebSocket 升级请求。

**解决：**
1. 关闭全局代理（Clash、V2Ray 等）
2. 或在 MCP 配置中添加 `--chromeArg=--no-proxy-server`
3. 修改配置后重启 MCP 服务

### 终端显示空白

**可能原因：**
1. WebSocket 未连接 — 检查控制台错误
2. `get token failed` — 认证问题，需重新登录
3. WebGL 上下文丢失 — 刷新页面

### 命令未被接收

**解决：**
1. 确保 textarea 获得焦点：`document.querySelector('.xterm-helper-textarea').focus()`
2. 文本输入用 `document.execCommand('insertText', false, char)`
3. 特殊按键用 `KeyboardEvent` 的 `keydown` 类型
4. **不要**使用 `InputEvent` 或 `compositionstart/end` — xterm.js 不处理这些事件

### Chrome MCP 浏览器无响应

```bash
pkill -f "chrome-devtools-mcp/chrome-profile"
# 等待 2 秒后重试 MCP 操作
```

---

## 8. 架构说明

| 组件 | 技术 |
|------|------|
| 前端 | React + xterm.js (v5+) + WebGL 渲染器 |
| WebSocket | `wss://console.d.pjlab.org.cn/ecp/pod/terminal?...`，子协议 `channel.k8s.io` |
| 认证 | HIGGS SSO，Bearer JWT token，加密存储在 localStorage |
| 心跳 | 15 秒间隔 ping/pong |
| 微前端 | ECP 应用作为 qiankun 微应用加载 |

### Canvas 索引

- `canvas[0]` — 链接层覆盖层
- `canvas[1]` — 主 WebGL 渲染 Canvas（截图用这个）

### 关键配置路径

- MCP 配置：`~/Library/Application Support/Trae CN/User/mcp.json`
- Chrome Profile：`~/.cache/chrome-devtools-mcp/chrome-profile/`
