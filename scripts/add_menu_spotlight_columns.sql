-- Home / kitchen spotlight flags on menu_items (run once per environment).
ALTER TABLE menu_items
  ADD COLUMN IF NOT EXISTS chef_special BOOLEAN NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS todays_special BOOLEAN NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS must_try BOOLEAN NOT NULL DEFAULT false;
