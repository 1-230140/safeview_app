create extension if not exists pgcrypto;

create table if not exists public.users (
  id uuid primary key default gen_random_uuid(),
  username text unique not null,
  role text not null default 'operator',
  password_hash text not null,
  created_at timestamptz not null default now()
);

create table if not exists public.scans (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references public.users(id) on delete set null,
  source text not null check (source in ('live', 'photo', 'import')),
  verdict text not null check (verdict in ('SAFE', 'WARNING', 'UNSAFE')),
  confidence text,
  risk_score int check (risk_score between 0 and 10),
  summary text,
  recommendation text,
  provider_used text not null default 'yolo-local',
  created_at timestamptz not null default now()
);

create table if not exists public.findings (
  id bigint generated always as identity primary key,
  scan_id uuid not null references public.scans(id) on delete cascade,
  level text not null check (level in ('safe', 'warn', 'unsafe')),
  text text not null,
  created_at timestamptz not null default now()
);

alter table public.users enable row level security;
alter table public.scans enable row level security;
alter table public.findings enable row level security;

drop policy if exists "anon read users for app login" on public.users;
drop policy if exists "anon insert scans" on public.scans;
drop policy if exists "anon read scans" on public.scans;
drop policy if exists "anon insert findings" on public.findings;
drop policy if exists "anon read findings" on public.findings;

create policy "anon read users for app login" on public.users for select to anon using (true);
create policy "anon insert scans" on public.scans for insert to anon with check (true);
create policy "anon read scans" on public.scans for select to anon using (true);
create policy "anon insert findings" on public.findings for insert to anon with check (true);
create policy "anon read findings" on public.findings for select to anon using (true);

insert into public.users (username, role, password_hash)
values (
  'admin',
  'admin',
  encode(digest('safeview2025', 'sha256'), 'hex')
)
on conflict (username) do update set
  role = excluded.role,
  password_hash = excluded.password_hash;
