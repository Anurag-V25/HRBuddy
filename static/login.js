(function(){
  const emailEl   = document.getElementById('email');
  const pwEl      = document.getElementById('password');
  const remember  = document.getElementById('remember');
  const loginBtn  = document.getElementById('loginBtn');
  const togglePw  = document.getElementById('togglePw');
  const healthBtn = document.getElementById('healthBtn');
  const toast     = document.getElementById('toast');
  const statusDot = document.getElementById('statusDot');
  const excelPath = document.getElementById('excelPath');
  const themeBtn  = document.getElementById('themeBtn');

  // Theme toggle & persist
  const THEME_KEY = 'hrbuddy_theme';
  const savedTheme = localStorage.getItem(THEME_KEY) || 'dark';
  if(savedTheme === 'light') document.body.classList.add('light');
  themeBtn.addEventListener('click', ()=>{
    document.body.classList.toggle('light');
    localStorage.setItem(THEME_KEY, document.body.classList.contains('light') ? 'light' : 'dark');
  });

  // Remember me (email)
  const REMEMBER_KEY = 'hrbuddy_email';
  const savedEmail = localStorage.getItem(REMEMBER_KEY);
  if(savedEmail){ emailEl.value = savedEmail; remember.checked = true; }

  function showToast(msg, ok=false){
    toast.textContent = msg;
    toast.classList.toggle('success', !!ok);
    toast.hidden = false;
    setTimeout(()=> toast.hidden = true, 2500);
  }

  togglePw.addEventListener('click', ()=>{
    pwEl.type = pwEl.type === 'password' ? 'text' : 'password';
  });

  async function callHealth(){
    try{
      const res = await fetch('/healthz');
      const data = await res.json();
      if(data.has_excel){ statusDot.classList.remove('red'); statusDot.classList.add('green'); }
      else { statusDot.classList.remove('green'); statusDot.classList.add('red'); }
      if(data.excel_path){ excelPath.textContent = data.excel_path; }
    }catch(e){ /* ignore */ }
  }
  healthBtn.addEventListener('click', ()=>{ callHealth(); showToast('Checked system status', true); });
  callHealth();

  async function doLogin(){
    const email = (emailEl.value||'').trim();
    const password = (pwEl.value||'').trim();
    if(!email || !password){ showToast('Please enter email and password'); return; }

    loginBtn.disabled = true; loginBtn.textContent = 'Signing in…';
    try{
      const res = await fetch('/auth/login', {
        method: 'POST',
        headers: { 'Content-Type':'application/json' },
        body: JSON.stringify({ email, password })
      });
      const data = await res.json();
      if(!res.ok){ throw new Error(data.error || 'Login failed'); }

      // Save email if requested
      if(remember.checked) localStorage.setItem(REMEMBER_KEY, email);
      else localStorage.removeItem(REMEMBER_KEY);

      // ✅ Save JWT so /chat can use it
      if(data.token){
        localStorage.setItem('hrbuddy_token', data.token);
      } else {
        throw new Error('No token returned from server');
      }

      showToast('Signed in', true);

  // Redirect using route_url from backend
  const dest = data.route_url || '/chat';
  setTimeout(()=> { window.location.href = dest; }, 700);

    }catch(err){
      showToast(err.message || String(err));
    }finally{
      loginBtn.disabled = false; loginBtn.textContent = 'Sign in';
    }
  }

  loginBtn.addEventListener('click', doLogin);
  document.getElementById('loginForm').addEventListener('keypress', (e)=>{
    if(e.key === 'Enter'){ doLogin(); }
  });
})();
